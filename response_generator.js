const TemplateMatcher = require('./template_matcher');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const axios = require('axios');
const levenshtein = require('fast-levenshtein');

class ResponseGenerator {
    constructor(knowledgePath = 'knowledge.json', trainingDataPath = 'training_data.json') {
        this.matcher = new TemplateMatcher(knowledgePath, trainingDataPath);
        this.model = null;
        this.vocab = {};
        this.conversations = {};
        this.trainingDataPath = trainingDataPath;
        this.trainingData = [];
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

                if (!Array.isArray(this.trainingData)) {
                    console.warn("⚠️ Warning: training_data.json is not an array. Resetting.");
                    this.trainingData = [];
                }
            } catch (error) {
                console.error("❌ Error loading training data:", error);
                this.trainingData = [];
            }
        } else {
            this.trainingData = [];
        }
    }

    async generateResponse(inputText) {
        this.learnInBackground(inputText);

        // Find a similar past conversation
        let closestMatch = this.findClosestMatch(inputText);
        if (closestMatch) {
            let refinedResponse = this.refineResponse(closestMatch.response, inputText);
            this.updateTrainingData(inputText, refinedResponse, true);
            return refinedResponse;
        }

        // Try template matching
        const { bestMatch, score } = this.matcher.findBestTemplate(inputText);
        if (bestMatch && score < 3) {
            return bestMatch.output;
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
        const unknownWords = words.filter(word => !this.vocab[word]);

        if (unknownWords.length > 0) {
            console.log("📚 Learning new words in background:", unknownWords);

            for (const word of unknownWords) {
                // Check if the word already exists in training data
                const existingEntry = this.trainingData.find(entry => 
                    entry.input.toLowerCase() === word.toLowerCase()
                );

                if (existingEntry) {
                    console.log(`ℹ️ Word '${word}' already exists in training data, skipping...`);
                    this.vocab[word] = Object.keys(this.vocab).length + 1; // Still add to vocab
                    continue;
                }

                if (this.vocab[word]) continue; // Skip if already in vocab
                this.vocab[word] = Object.keys(this.vocab).length + 1; // Assign an index

                try {
                    const definition = await this.getWikipediaInfo(word);
                    if (definition) {
                        // Add new entry instead of updating existing one
                        this.addNewTrainingEntry(word, definition);
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
                console.warn(`⚠️ Wikipedia returned a disambiguation (not available) page for ${word}. Skipping.`);
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

    addNewTrainingEntry(input, response) {
        // Check if entry already exists
        const exists = this.trainingData.some(entry => 
            entry.input.toLowerCase() === input.toLowerCase()
        );

        if (!exists) {
            this.trainingData.push({ input, response });
            try {
                fs.writeFileSync(this.trainingDataPath, JSON.stringify(this.trainingData, null, 2));
                console.log(`📚 Added new training entry: ${input} -> ${response}`);
            } catch (error) {
                console.error("❌ Failed to write training data:", error);
            }
        } else {
            console.log(`ℹ️ Training entry for '${input}' already exists, preserving existing data`);
        }
    }

    findClosestMatch(inputText) {
        if (!Array.isArray(this.trainingData) || this.trainingData.length === 0) {
            return null;
        }

        let bestMatch = null;
        let lowestDistance = Infinity;

        for (const entry of this.trainingData) {
            const distance = levenshtein.get(inputText.toLowerCase(), entry.input.toLowerCase());
            if (distance < lowestDistance) {
                lowestDistance = distance;
                bestMatch = entry;
            }
        }

        return lowestDistance < 10 ? bestMatch : null;
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
        // Only update if it's a refinement or the entry doesn't exist
        const existingEntryIndex = this.trainingData.findIndex(entry => 
            levenshtein.get(input.toLowerCase(), entry.input.toLowerCase()) < 3
        );

        if (existingEntryIndex === -1) {
            // Entry doesn't exist, add it
            this.trainingData.push({ input, response });
        } else if (isRefinement) {
            // Only update if it's a refinement
            this.trainingData[existingEntryIndex].response = response;
        }

        try {
            fs.writeFileSync(this.trainingDataPath, JSON.stringify(this.trainingData, null, 2));
            console.log(`📚 ${existingEntryIndex === -1 ? 'Added new' : 'Updated'} training data: ${input} -> ${response}`);
        } catch (error) {
            console.error("❌ Failed to write training data:", error);
        }
    }
}

module.exports = ResponseGenerator;
