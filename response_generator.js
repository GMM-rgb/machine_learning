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
            lastTrainingDate: '2025-02-04 04:46:46'
        };
        this.currentUser = 'GMM-rgb';
        this.currentDateTime = '2025-02-04 04:46:46';
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
        if (!inputText || typeof inputText !== 'string') {
            return "I'm not sure how to respond to that.";
        }

        try {
            await this.learnInBackground(inputText);

            const context = this.buildResponseContext(inputText);
            
            // Try to find a similar conversation with context
            let closestMatch = this.findClosestMatch(inputText, context);
            if (closestMatch && closestMatch.confidence > 0.7) {
                let refinedResponse = this.refineResponse(closestMatch.output, inputText, context);
                await this.updateTrainingData(inputText, refinedResponse, true);
                return refinedResponse;
            }

            // Try template matching with context
            const { bestMatch, score } = this.matcher.findBestTemplate(inputText);
            if (bestMatch && score < 3) {
                let response = this.enhanceResponse(bestMatch.output, context);
                await this.updateTrainingData(inputText, response, false);
                return response;
            }

            // Use AI model as a last resort
            if (this.model) {
                try {
                    let response = await this.generateModelResponse(inputText);
                    if (response) {
                        response = this.enhanceResponse(response, context);
                        await this.updateTrainingData(inputText, response, false);
                        return response;
                    }
                } catch (error) {
                    console.error("❌ Error generating model response:", error);
                }
            }

            return "I'm not sure how to respond to that.";
        } catch (error) {
            console.error("❌ Error in generateResponse:", error);
            return "I'm having trouble processing that message.";
        }
    }

    buildResponseContext(inputText) {
        if (!inputText || typeof inputText !== 'string') {
            return {
                definitions: {},
                relatedConversations: [],
                keyTerms: new Set()
            };
        }

        try {
            const words = inputText.toLowerCase().split(/\s+/);
            const context = {
                definitions: {},
                relatedConversations: [],
                keyTerms: new Set()
            };

            // Gather definitions for context
            for (const word of words) {
                if (!word) continue;
                const definition = this.findDefinition(word);
                if (definition) {
                    context.definitions[word] = definition;
                    context.keyTerms.add(word);
                }
            }

            // Find related conversations
            context.relatedConversations = this.findRelatedConversations(inputText, Array.from(context.keyTerms));

            return context;
        } catch (error) {
            console.error("❌ Error building context:", error);
            return {
                definitions: {},
                relatedConversations: [],
                keyTerms: new Set()
            };
        }
    }

    async generateModelResponse(inputText) {
        if (!inputText || typeof inputText !== 'string' || !this.model) {
            return null;
        }

        try {
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
        } catch (error) {
            console.error("❌ Error in generateModelResponse:", error);
            return null;
        }
    }

    findRelatedConversations(inputText, keyTerms) {
        if (!inputText || !this.trainingData.conversations) {
            return [];
        }

        try {
            return this.trainingData.conversations
                .filter(conv => {
                    if (!conv || !conv.input || !conv.output) return false;

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
                .slice(0, 5);
        } catch (error) {
            console.error("❌ Error finding related conversations:", error);
            return [];
        }
    }

    async learnInBackground(inputText) {
        if (!inputText || typeof inputText !== 'string') return;

        try {
            const words = inputText.toLowerCase().split(/\s+/);
            const unknownWords = words.filter(word => !this.findDefinition(word));

            if (unknownWords.length > 0) {
                console.log("📚 Learning new words in background:", unknownWords);

                for (const word of unknownWords) {
                    if (!word) continue;

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
        } catch (error) {
            console.error("❌ Error in learnInBackground:", error);
        }
    }

    async getWikipediaInfo(word) {
        if (!word || typeof word !== 'string') return null;

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
        if (!word || typeof word !== 'string' || !this.trainingData.definitions) {
            return null;
        }

        try {
            const definition = this.trainingData.definitions.find(
                def => def && def.word && word && 
                def.word.toLowerCase() === word.toLowerCase()
            );
            return definition ? definition.definition : null;
        } catch (error) {
            console.error("❌ Error finding definition:", error);
            return null;
        }
    }

    async addNewDefinition(word, definition) {
        if (!word || !definition || !this.trainingData.definitions) return;

        try {
            const existingDefIndex = this.trainingData.definitions.findIndex(
                def => def && def.word && def.word.toLowerCase() === word.toLowerCase()
            );

            if (existingDefIndex === -1) {
                this.trainingData.definitions.push({
                    word: word,
                    definition: definition
                });
                await this.saveTrainingData();
                console.log(`📚 Added new definition: ${word} -> ${definition}`);
            }
        } catch (error) {
            console.error("❌ Error adding new definition:", error);
        }
    }

    findClosestMatch(inputText, context) {
        if (!inputText || !this.trainingData.conversations || this.trainingData.conversations.length === 0) {
            return null;
        }

        try {
            let bestMatch = null;
            let highestConfidence = 0;

            // Consider both direct matches and context-enhanced matches
            for (const conversation of this.trainingData.conversations) {
                if (!conversation || !conversation.input) continue;

                let confidence = 0;
                
                // Basic text similarity
                const distance = levenshtein.get(inputText.toLowerCase(), conversation.input.toLowerCase());
                const maxLength = Math.max(inputText.length, conversation.input.length);
                confidence = 1 - (distance / maxLength);

                // Boost confidence if conversation contains key terms from context
                if (context && context.keyTerms && context.keyTerms.size > 0) {
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
        } catch (error) {
            console.error("❌ Error finding closest match:", error);
            return null;
        }
    }

    enhanceResponse(response, context) {
        if (!response || typeof response !== 'string') {
            return response;
        }

        try {
            let enhancedResponse = response;

            // Use context to make response more relevant
            if (context && context.relatedConversations && context.relatedConversations.length > 0) {
                const relevantPhrases = context.relatedConversations
                    .map(conv => conv.output)
                    .filter(output => output && output !== response);

                // Incorporate relevant phrases if they exist
                if (relevantPhrases.length > 0) {
                    const relevantPhrase = relevantPhrases[0]; // Use the most relevant one
                    if (response.length < 50) { // Only enhance short responses
                        enhancedResponse = `${response} ${relevantPhrase}`;
                    }
                }
            }

            return enhancedResponse;
        } catch (error) {
            console.error("❌ Error enhancing response:", error);
            return response;
        }
    }

    refineResponse(existingResponse, inputText, context) {
        if (!existingResponse || !inputText) {
            return existingResponse;
        }

        try {
            const inputWords = inputText.toLowerCase().split(/\s+/);
            const responseWords = existingResponse.toLowerCase().split(/\s+/);

            let refinedResponse = responseWords.map(word => {
                if (!word) return word;
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
        } catch (error) {
            console.error("❌ Error refining response:", error);
            return existingResponse;
        }
    }

    findSimilarWord(word, inputWords) {
        if (!word || !inputWords || !Array.isArray(inputWords)) {
            return word;
        }

        try {
            let closestWord = word;
            let lowestDistance = Infinity;

            for (const inputWord of inputWords) {
                if (!inputWord) continue;
                const distance = levenshtein.get(word, inputWord);
                if (distance < lowestDistance) {
                    lowestDistance = distance;
                    closestWord = inputWord;
                }
            }

            return lowestDistance <= 2 ? closestWord : word;
        } catch (error) {
            console.error("❌ Error finding similar word:", error);
            return word;
        }
    }

    async updateTrainingData(input, response, isRefinement) {
        if (!input || !response || !this.trainingData.conversations) return;

        try {
            const existingIndex = this.trainingData.conversations.findIndex(conv => 
                conv && conv.input && conv.input.toLowerCase() === input.toLowerCase()
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
            await this.saveTrainingData();
        } catch (error) {
            console.error("❌ Error updating training data:", error);
        }
    }

    async saveTrainingData() {
        try {
            // Update last training date before saving
            this.trainingData.lastTrainingDate = '2025-02-04 04:48:22';
            
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
