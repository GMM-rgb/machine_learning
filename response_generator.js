const TemplateMatcher = require('./template_matcher');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const axios = require('axios');
const levenshtein = require('fast-levenshtein');

// TensorFlow setup - Ensure this is only done once
if (!global.tfSetup) {
    global.tfSetup = true;
    // Any additional TensorFlow setup code
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
            lastTrainingDate: ""
        };
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

        // First, check for definitions
        const words = inputText.toLowerCase().split(/\s+/);
        for (const word of words) {
            const definition = this.findDefinition(word);
            if (definition) {
                return definition;
            }
        }

        // Try to find a similar conversation
        let closestMatch = this.findClosestMatch(inputText);
        if (closestMatch && closestMatch.confidence > 0.7) {
            let refinedResponse = this.refineResponse(closestMatch.output, inputText);
            this.updateTrainingData(inputText, refinedResponse, true);
            return refinedResponse;
        }

        // Try template matching
        const { bestMatch, score } = this.matcher.findBestTemplate(inputText);
        if (bestMatch && score < 3) {
            let response = bestMatch.output;
            this.updateTrainingData(inputText, response, false);
            return response;
        }

        // Use AI model as a last resort
        if (this.model) {
            try {
                let response = await this.generateModelResponse(inputText);
                if (response) {
                    this.updateTrainingData(inputText, response, false);
                    return response;
                }
            } catch (error) {
                console.error("❌ Error generating model response:", error);
            }
        }

        return "I'm not sure how to respond to that.";
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

    findClosestMatch(inputText) {
        if (!this.trainingData.conversations || this.trainingData.conversations.length === 0) {
            return null;
        }

        let bestMatch = null;
        let highestConfidence = 0;

        for (const conversation of this.trainingData.conversations) {
            const distance = levenshtein.get(inputText.toLowerCase(), conversation.input.toLowerCase());
            const maxLength = Math.max(inputText.length, conversation.input.length);
            const confidence = 1 - (distance / maxLength);

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

    refineResponse(existingResponse, inputText) {
        const inputWords = inputText.toLowerCase().split(/\s+/);
        const responseWords = existingResponse.toLowerCase().split(/\s+/);

        let refinedResponse = responseWords.map(word => {
            if (inputWords.includes(word)) {
                return word;
            }
            return this.findSimilarWord(word, inputWords);
        }).join(' ');

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
            timestamp: new Date(this.currentDateTime).toISOString()
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
