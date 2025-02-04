const TemplateMatcher = require('./template_matcher');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const axios = require('axios');
const levenshtein = require('fast-levenshtein');

// TensorFlow setup - Ensure this is only done once
if (!global.tfSetup) {
    global.tfSetup = true;
}

class ResponseGenerator {
    constructor(knowledgePath = 'knowledge.json', trainingDataPath = 'training_data.json') {
        this.matcher = new TemplateMatcher(knowledgePath, trainingDataPath);
        this.model = null;
        this.vocab = {};
        this.trainingDataPath = trainingDataPath;
        this.trainingData = {
            conversations: [],
            definitions: [],
            vocabulary: {},
            lastTrainingDate: '2025-02-04 02:34:09'
        };
        this.currentUser = 'GMM-rgb';
        this.currentDateTime = '2025-02-04 02:34:09';
        this.loadModel();
        this.loadTrainingData();
    }

    async loadModel() {
        try {
            this.model = await tf.loadLayersModel('file://D:/machine_learning/model.json');
            console.log("✅ Model loaded successfully in ResponseGenerator");
        } catch (error) {
            console.error("❌ Error loading model in ResponseGenerator:", error);
        }
    }

    loadTrainingData() {
        if (fs.existsSync(this.trainingDataPath)) {
            try {
                const data = fs.readFileSync(this.trainingDataPath, 'utf8');
                this.trainingData = JSON.parse(data);

                // Initialize if structure is missing
                if (!this.trainingData.conversations) {
                    this.trainingData.conversations = [];
                }
                if (!this.trainingData.definitions) {
                    this.trainingData.definitions = [];
                }
                if (!this.trainingData.vocabulary) {
                    this.trainingData.vocabulary = {};
                }
                if (!this.trainingData.lastTrainingDate) {
                    this.trainingData.lastTrainingDate = this.currentDateTime;
                }

                // Load vocabulary into vocab object
                this.vocab = { ...this.trainingData.vocabulary };
            } catch (error) {
                console.error("❌ Error loading training data:", error);
                this.initializeEmptyTrainingData();
            }
        } else {
            this.initializeEmptyTrainingData();
        }
    }

    initializeEmptyTrainingData() {
        this.trainingData = {
            conversations: [],
            definitions: [],
            vocabulary: {},
            lastTrainingDate: this.currentDateTime
        };
    }
    
    async generateResponse(inputText) {
        await this.learnInBackground(inputText);

        // First, try to find similar conversations and context
        const context = this.buildResponseContext(inputText);
        
        // Try to find a similar conversation with context
        let closestMatch = this.findClosestMatch(inputText, context);
        if (closestMatch && closestMatch.confidence > 0.7) {
            let refinedResponse = this.refineResponse(closestMatch.output, inputText, context);
            this.updateTrainingData(inputText, refinedResponse, true);
            return refinedResponse;
        }

        // Try template matching with context
        const { bestMatch, score } = this.matcher.findBestTemplate(inputText);
        if (bestMatch && score < 3) {
            let response = this.enhanceResponse(bestMatch.output, context);
            this.updateTrainingData(inputText, response, false);
            return response;
        }

        // Use AI model as a last resort
        if (this.model) {
            try {
                let response = await this.generateModelResponse(inputText);
                if (response) {
                    response = this.enhanceResponse(response, context);
                    this.updateTrainingData(inputText, response, false);
                    return response;
                }
            } catch (error) {
                console.error("❌ Error generating model response:", error);
            }
        }

        return "I'm not sure how to respond to that.";
    }

    buildResponseContext(inputText) {
        const words = inputText.toLowerCase().split(/\s+/);
        const context = {
            definitions: {},
            relatedConversations: [],
            keyTerms: new Set()
        };

        // Gather definitions for context
        for (const word of words) {
            const definition = this.findDefinition(word);
            if (definition) {
                context.definitions[word] = definition;
                context.keyTerms.add(word);
            }
        }

        // Find related conversations
        context.relatedConversations = this.findRelatedConversations(inputText, Array.from(context.keyTerms));

        return context;
    }

    async generateModelResponse(inputText) {
        const words = inputText.toLowerCase().split(/\s+/);
        const inputSequence = words.map(word => this.vocab[word] || 0);
        const inputTensor = tf.tensor2d([inputSequence], [1, inputSequence.length]);
        const prediction = this.model.predict(inputTensor);
        const probabilities = await prediction.array();

        const responseIndices = probabilities[0]
            .map((prob, index) => ({ prob, index }))
            .sort((a, b) => b.prob - a.prob)
            .slice(0, 20)
            .map(item => item.index);

        const reverseVocab = Object.fromEntries(
            Object.entries(this.vocab).map(([word, index]) => [index, word])
        );

        return responseIndices.map(index => reverseVocab[index]).filter(word => word).join(' ');
    }

    findRelatedConversations(inputText, keyTerms) {
        return this.trainingData.conversations
            .filter(conv => {
                // Check if conversation contains any key terms
                const containsKeyTerm = keyTerms.some(term => 
                    conv.input.toLowerCase().includes(term) || 
                    conv.output.toLowerCase().includes(term)
                );

                // Calculate similarity
                const inputSimilarity = levenshtein.get(
                    inputText.toLowerCase(), 
                    conv.input.toLowerCase()
                );

                return containsKeyTerm || inputSimilarity < 10;
            })
            .map(conv => ({
                ...conv,
                similarity: 1 - (levenshtein.get(inputText.toLowerCase(), conv.input.toLowerCase()) / 
                              Math.max(inputText.length, conv.input.length))
            }))
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, 5); // Get top 5 related conversations
    }

    async learnInBackground(inputText) {
        const words = inputText.toLowerCase().split(/\s+/);
        const unknownWords = words.filter(word => !this.findDefinition(word));

        if (unknownWords.length > 0) {
            console.log("📚 Learning new words in background:", unknownWords);

            for (const word of unknownWords) {
                // Skip if word already exists in definitions
                if (this.findDefinition(word)) {
                    console.log(`ℹ️ Word '${word}' already exists in definitions, skipping...`);
                    continue;
                }

                try {
                    const definition = await this.getWikipediaInfo(word);
                    if (definition) {
                        await this.addNewDefinition(word, definition);
                        console.log(`✅ Learned new word: ${word} -> ${definition}`);
                    } else {
                        console.log(`⚠️ No Wikipedia definition found for word: ${word}`);
                    }
                } catch (error) {
                    console.error(`❌ Error learning word '${word}':`, error);
                }
            }
        }
    }

    async getWikipediaInfo(word) {
        try {
            const response = await axios.get(`https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(word)}`);

            if (response.data.type === "disambiguation") {
                console.warn(`⚠️ Wikipedia returned a disambiguation page for ${word}. Skipping.`);
                return null;
            }

            if (response.data.extract) {
                return response.data.extract.split('. ')[0]; // Take the first sentence
            } else {
                console.warn(`⚠️ Wikipedia has no summary for ${word}`);
                return null;
            }
        } catch (error) {
            console.error(`❌ Wikipedia fetch failed for ${word}:`, error.response?.status || error.message);
            return null;
        }
    }

    findDefinition(word) {
        const definition = this.trainingData.definitions.find(
            def => def.word.toLowerCase() === word.toLowerCase()
        );
        return definition ? definition.definition : null;
    }

    async addNewDefinition(word, definition) {
        const existingDefIndex = this.trainingData.definitions.findIndex(
            def => def.word.toLowerCase() === word.toLowerCase()
        );

        if (existingDefIndex === -1) {
            this.trainingData.definitions.push({
                word: word,
                definition: definition
            });
            await this.saveTrainingData();
            console.log(`📚 Added new definition: ${word} -> ${definition}`);
        }
    }

    findClosestMatch(inputText, context) {
        if (!this.trainingData.conversations || this.trainingData.conversations.length === 0) {
            return null;
        }

        let bestMatch = null;
        let highestConfidence = 0;

        // Consider both direct matches and context-enhanced matches
        for (const conversation of this.trainingData.conversations) {
            let confidence = 0;
            
            // Basic text similarity
            const distance = levenshtein.get(inputText.toLowerCase(), conversation.input.toLowerCase());
            const maxLength = Math.max(inputText.length, conversation.input.length);
            confidence = 1 - (distance / maxLength);

            // Boost confidence if conversation contains key terms from context
            if (context.keyTerms.size > 0) {
                const keyTermBoost = Array.from(context.keyTerms).reduce((boost, term) => {
                    if (conversation.input.toLowerCase().includes(term) || 
                        conversation.output.toLowerCase().includes(term)) {
                        return boost + 0.1; // Boost for each matching key term
                    }
                    return boost;
                }, 0);
                confidence += keyTermBoost;
            }

            if (confidence > highestConfidence) {
                highestConfidence = confidence;
                bestMatch = {
                    ...conversation,
                    confidence
                };
            }
        }

        return bestMatch;
    }

    enhanceResponse(response, context) {
        // Don't return definitions directly, but use them to enhance the response
        let enhancedResponse = response;

        // Use context to make response more relevant
        if (context.relatedConversations.length > 0) {
            const relevantPhrases = context.relatedConversations
                .map(conv => conv.output)
                .filter(output => output !== response);

            // Incorporate relevant phrases if they exist
            if (relevantPhrases.length > 0) {
                const relevantPhrase = relevantPhrases[0]; // Use the most relevant one
                if (response.length < 50) { // Only enhance short responses
                    enhancedResponse = `${response} ${relevantPhrase}`;
                }
            }
        }

        return enhancedResponse;
    }

    refineResponse(existingResponse, inputText, context) {
        const inputWords = inputText.toLowerCase().split(/\s+/);
        const responseWords = existingResponse.toLowerCase().split(/\s+/);

        let refinedResponse = responseWords.map(word => {
            if (inputWords.includes(word)) {
                return word;
            }
            return this.findSimilarWord(word, inputWords);
        }).join(' ');

        // Enhance with context if the response is too short or generic
        if (refinedResponse.split(' ').length < 5) {
            refinedResponse = this.enhanceResponse(refinedResponse, context);
        }

        return refinedResponse;
    }

    findSimilarWord(word, inputWords) {
        let closestWord = word;
        let lowestDistance = Infinity;

        for (const inputWord of inputWords) {
            const distance = levenshtein.get(word, inputWord);
            if (distance < lowestDistance) {
                lowestDistance = distance;
                closestWord = inputWord;
            }
        }

        return lowestDistance <= 2 ? closestWord : word;
    }

    updateTrainingData(input, response, isRefinement) {
        const existingIndex = this.trainingData.conversations.findIndex(conv => 
            conv.input.toLowerCase() === input.toLowerCase()
        );

        const newEntry = {
            input,
            output: response,
            timestamp: new Date(this.currentDateTime).toISOString(),
            user: this.currentUser
        };

        if (existingIndex === -1) {
            // New conversation
            this.trainingData.conversations.push(newEntry);
        } else if (isRefinement) {
            // Update existing conversation only if it's a refinement
            this.trainingData.conversations[existingIndex] = newEntry;
        }

        // Update last training date
        this.trainingData.lastTrainingDate = this.currentDateTime;
        this.saveTrainingData();
    }

    async saveTrainingData() {
        try {
            // Update last training date before saving
            this.trainingData.lastTrainingDate = this.currentDateTime;
            
            await fs.promises.writeFile(
                this.trainingDataPath,
                JSON.stringify(this.trainingData, null, 2)
            );
            console.log('📝 Training data saved successfully');
        } catch (error) {
            console.error('❌ Error saving training data:', error);
        }
    }
}

module.exports = ResponseGenerator;
