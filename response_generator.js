const TemplateMatcher = require('./template_matcher');
const tf = require('@tensorflow/tfjs-node-gpu');
const fs = require('fs');
const axios = require('axios');
const levenshtein = require('fast-levenshtein');
const { ifError } = require('assert');
const path = require('path');
const { isArray } = require('mathjs');
const readline = require('readline');
const chalk = require('chalk');  // This will work with chalk@4.1.2
const natural = require('natural');
const tokenizer = new natural.WordTokenizer();

// Global cache for model and responses
const globalCache = {
    model: null,
    responses: new Map(),
    embeddings: new Map(),
    lastUpdate: '2025-02-05 04:38:59'
};

class ResponseGenerator {
    constructor(knowledgePath = 'knowledge.json', trainingDataPath = 'training_data.json', modelPath = 'model/') {
        this.matcher = new TemplateMatcher(knowledgePath, trainingDataPath);
        this.modelPath = modelPath;
        this.vocab = {};
        this.trainingDataPath = trainingDataPath;
        this.trainingData = {
            conversations: [],
            definitions: [],
            vocabulary: {},
            lastTrainingDate: '2025-02-05 04:38:59'
        };
        this.currentUser = 'GMM-rgb';
        this.currentDateTime = '2025-02-05 04:38:59';
        this.modelCache = new Map();
        this.responseCache = new Map();
        
        // Initialize NLP tools
        this.tokenizer = new natural.WordTokenizer();
        this.sentenceTokenizer = new natural.SentenceTokenizer();
        this.tfidf = new natural.TfIdf();

        // Initialize immediately
        this.initialize();
    }

    async initialize() {
        await this.loadModel();
        await this.loadTrainingData();
        this.setupModelCache();
    }

    setupModelCache() {
        // Setup LRU cache for model outputs
        this.modelCache.maxSize = 1000;
        this.responseCache.maxSize = 500;

        // Clean old cache entries periodically
        setInterval(() => {
            const now = new Date().getTime();
            for (const [key, value] of this.modelCache) {
                if (now - value.timestamp > 3600000) { // 1 hour
                    this.modelCache.delete(key);
                }
            }
        }, 900000); // Clean every 15 minutes
    }

    async loadModel() {
        if (globalCache.model) {
            console.log(chalk.green("✅ Using cached model"));
            this.model = globalCache.model;
            return;
        }

        try {
            // Try loading from memory first
            if (process.env.USE_MEMORY_CACHE === 'true' && globalCache.model) {
                this.model = globalCache.model;
                console.log(chalk.green("✅ Model loaded from memory cache"));
                return;
            }

            // Try loading saved model
            const modelFiles = await fs.promises.readdir(this.modelPath).catch(() => []);
            if (modelFiles.length > 0) {
                this.model = await tf.loadLayersModel(`file://${this.modelPath}/model.json`);
                globalCache.model = this.model;
                console.log(chalk.green("✅ Model loaded from file"));
                return;
            }

            // If no model exists, create a new one
            console.log(chalk.yellow("Creating new model..."));
            this.model = await this.createNewModel();
            globalCache.model = this.model;
            await this.model.save(`file://${this.modelPath}`);
            console.log(chalk.green("✅ New model created and saved"));

        } catch (error) {
            console.error(chalk.red("❌ Error loading model:"), error);
            // Create emergency backup model
            this.model = await this.createNewModel();
            globalCache.model = this.model;
        }
    }

    async createNewModel() {
        const model = tf.sequential();
        
        // Add layers for text processing
        model.add(tf.layers.embedding({
            inputDim: 10000,
            outputDim: 128,
            inputLength: 50
        }));
        
        model.add(tf.layers.lstm({
            units: 64,
            returnSequences: true
        }));
        
        model.add(tf.layers.lstm({
            units: 32
        }));
        
        model.add(tf.layers.dense({
            units: 64,
            activation: 'relu'
        }));
        
        model.add(tf.layers.dense({
            units: 32,
            activation: 'relu'
        }));
        
        model.add(tf.layers.dense({
            units: 16,
            activation: 'softmax'
        }));

        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        return model;
    }

    loadTrainingData() {
        if (fs.existsSync(this.trainingDataPath)) {
            try {
                const data = fs.readFileSync(this.trainingDataPath, 'utf8');
                this.trainingData = JSON.parse(data);

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

                this.vocab = { ...this.trainingData.vocabulary };
                this.initializeTFIDF();
                console.log(chalk.green("✅ Training data loaded"));
            } catch (error) {
                console.error(chalk.red("❌ Error loading data:"), error);
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

    initializeTFIDF() {
        this.tfidf = new natural.TfIdf();
        this.trainingData.conversations.forEach(conv => {
            if (conv && conv.input) {
                this.tfidf.addDocument(conv.input.toLowerCase());
            }
        });
    }

    async startConsoleInterface() {
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });

        console.log(chalk.cyan('\n=== Training Data Management ==='));
        console.log(chalk.yellow('Commands:'));
        console.log('1. view    - View data');
        console.log('2. add     - Add data');
        console.log('3. edit    - Edit data');
        console.log('4. delete  - Delete data');
        console.log('5. search  - Search data');
        console.log('6. test    - Test response');
        console.log('7. stats   - View stats');
        console.log('8. export  - Export data');
        console.log('9. exit    - Exit console');

        const handleCommand = async (command) => {
            switch(command.toLowerCase()) {
                case 'view':
                    await this.viewTrainingData();
                    break;
                case 'add':
                    await this.addTrainingDataConsole(rl);
                    break;
                case 'edit':
                    await this.editTrainingDataConsole(rl);
                    break;
                case 'delete':
                    await this.deleteTrainingDataConsole(rl);
                    break;
                case 'search':
                    await this.searchTrainingDataConsole(rl);
                    break;
                case 'test':
                    await this.testResponseGenerationConsole(rl);
                    break;
                case 'stats':
                    await this.viewStatistics();
                    break;
                case 'export':
                    await this.exportTrainingData();
                    break;
                case 'exit':
                    console.log(chalk.green('Goodbye!'));
                    rl.close();
                    return;
                default:
                    console.log(chalk.red('Invalid command'));
            }
            
            rl.question(chalk.cyan('\nEnter command: '), async (cmd) => {
                await handleCommand(cmd);
            });
        };

        rl.question(chalk.cyan('Enter command: '), async (cmd) => {
            await handleCommand(cmd);
        });
    }

    async viewTrainingData() {
        console.log(chalk.green('\nCurrent Training Data:'));
        if (this.trainingData.conversations.length === 0) {
            console.log(chalk.yellow('No data available.'));
            return;
        }

        this.trainingData.conversations.forEach((conv, index) => {
            console.log(chalk.yellow(`\n[${index + 1}]`));
            console.log(chalk.cyan('Input:     ') + conv.input);
            console.log(chalk.cyan('Output:    ') + conv.output);
            console.log(chalk.cyan('Timestamp: ') + conv.timestamp);
            console.log(chalk.cyan('User:      ') + conv.user);
        });
    }

    async addTrainingDataConsole(rl) {
        const input = await new Promise(resolve => {
            rl.question(chalk.cyan('Enter input: '), resolve);
        });

        const output = await new Promise(resolve => {
            rl.question(chalk.cyan('Enter output: '), resolve);
        });

        await this.updateTrainingData(
            this.properlyCapitalize(input),
            this.properlyCapitalize(output),
            false
        );
        console.log(chalk.green('✅ Data added'));
    }

    async editTrainingDataConsole(rl) {
        await this.viewTrainingData();
        const index = await new Promise(resolve => {
            rl.question(chalk.cyan('Enter index to edit: '), resolve);
        });

        const idx = parseInt(index) - 1;
        if (idx >= 0 && idx < this.trainingData.conversations.length) {
            const currentEntry = this.trainingData.conversations[idx];
            console.log(chalk.yellow('\nCurrent values:'));
            console.log(`Input: ${currentEntry.input}`);
            console.log(`Output: ${currentEntry.output}`);

            const input = await new Promise(resolve => {
                rl.question(chalk.cyan('\nNew input (Enter to keep): '), resolve);
            });

            const output = await new Promise(resolve => {
                rl.question(chalk.cyan('New output (Enter to keep): '), resolve);
            });

            this.trainingData.conversations[idx] = {
                input: input.trim() ? this.properlyCapitalize(input) : currentEntry.input,
                output: output.trim() ? this.properlyCapitalize(output) : currentEntry.output,
                timestamp: new Date().toISOString(),
                user: this.currentUser
            };

            await this.saveTrainingData();
            console.log(chalk.green('✅ Updated'));
        } else {
            console.log(chalk.red('❌ Invalid index'));
        }
    }

    async deleteTrainingDataConsole(rl) {
        await this.viewTrainingData();
        const index = await new Promise(resolve => {
            rl.question(chalk.cyan('Enter index to delete: '), resolve);
        });

        const idx = parseInt(index) - 1;
        if (idx >= 0 && idx < this.trainingData.conversations.length) {
            const confirm = await new Promise(resolve => {
                rl.question(chalk.yellow('Confirm delete? (y/n): '), resolve);
            });

            if (confirm.toLowerCase() === 'y') {
                this.trainingData.conversations.splice(idx, 1);
                await this.saveTrainingData();
                console.log(chalk.green('✅ Deleted'));
            } else {
                console.log(chalk.yellow('Cancelled'));
            }
        } else {
            console.log(chalk.red('❌ Invalid index'));
        }
    }

    async searchTrainingDataConsole(rl) {
        const searchTerm = await new Promise(resolve => {
            rl.question(chalk.cyan('Search term: '), resolve);
        });

        const results = this.trainingData.conversations.filter(conv => 
            conv.input.toLowerCase().includes(searchTerm.toLowerCase()) ||
            conv.output.toLowerCase().includes(searchTerm.toLowerCase())
        );

        if (results.length > 0) {
            console.log(chalk.green(`\nFound ${results.length} matches:`));
            results.forEach((conv, index) => {
                console.log(chalk.yellow(`\n[${index + 1}]`));
                console.log(chalk.cyan('Input:  ') + this.highlightText(conv.input, searchTerm));
                console.log(chalk.cyan('Output: ') + this.highlightText(conv.output, searchTerm));
            });
        } else {
            console.log(chalk.yellow('No matches'));
        }
    }

    async testResponseGenerationConsole(rl) {
        const input = await new Promise(resolve => {
            rl.question(chalk.cyan('Test input: '), resolve);
        });

        console.log(chalk.yellow('\nGenerating...'));
        const possibilities = await this.generateEnhancedResponse(input);

        if (possibilities && possibilities.length > 0) {
            console.log(chalk.green('\nResponses:'));
            possibilities.forEach((p, index) => {
                console.log(chalk.yellow(`\n[${index + 1}] ${(p.confidence * 100).toFixed(2)}%`));
                console.log(chalk.cyan('Response: ') + p.response);
                console.log(chalk.cyan('Source:   ') + p.source);
            });
        } else {
            console.log(chalk.red('No responses'));
        }
    }

    async viewStatistics() {
        console.log(chalk.green('\nStatistics:'));
        console.log(chalk.cyan('Conversations:  ') + this.trainingData.conversations.length);
        console.log(chalk.cyan('Definitions:    ') + this.trainingData.definitions.length);
        console.log(chalk.cyan('Vocabulary:     ') + Object.keys(this.trainingData.vocabulary).length);
        console.log(chalk.cyan('Last Training:  ') + this.trainingData.lastTrainingDate);

        const avgLength = this.trainingData.conversations.reduce((acc, conv) => 
            acc + (conv.output ? conv.output.length : 0), 0) / 
            (this.trainingData.conversations.length || 1);

        console.log(chalk.cyan('Avg Response:   ') + avgLength.toFixed(2) + ' chars');
    }

async exportTrainingData() {
        const exportPath = `training_data_export_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
        try {
            await fs.promises.writeFile(
                exportPath,
                JSON.stringify(this.trainingData, null, 2)
            );
            console.log(chalk.green(`✅ Exported to ${exportPath}`));
        } catch (error) {
            console.error(chalk.red('❌ Export error:'), error);
        }
    }

    highlightText(text, searchTerm) {
        if (!searchTerm) return text;
        const regex = new RegExp(searchTerm, 'gi');
        return text.replace(regex, match => chalk.bgYellow.black(match));
    }

    properlyCapitalize(text) {
        if (!text) return text;

        const sentences = this.sentenceTokenizer.tokenize(text);
        return sentences.map(sentence => {
            if (!sentence.trim()) return sentence;
            
            const specialWords = ['i', 'i\'m', 'i\'ll', 'i\'ve', 'i\'d'];
            
            return sentence.split(' ').map((word, index) => {
                if (index === 0 || specialWords.includes(word.toLowerCase())) {
                    return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
                }
                return word.toLowerCase();
            }).join(' ');
        }).join(' ');
    }

    async generateModelResponse(inputText) {
        // Check cache first
        const cacheKey = inputText.toLowerCase().trim();
        if (this.responseCache.has(cacheKey)) {
            const cached = this.responseCache.get(cacheKey);
            if (new Date().getTime() - cached.timestamp < 3600000) { // 1 hour cache
                return cached.response;
            }
        }

        try {
            // Prepare input
            const tokens = this.tokenizer.tokenize(inputText.toLowerCase());
            const paddedTokens = [...tokens.slice(0, 50), ...Array(Math.max(0, 50 - tokens.length)).fill(0)];
            
            // Convert to tensor
            const inputTensor = tf.tensor2d([paddedTokens], [1, 50]);
            
            // Get prediction
            const prediction = this.model.predict(inputTensor);
            const responseIndex = tf.argMax(prediction, 1).dataSync()[0];
            
            // Get response from training data
            let response = "I'm learning to respond to that.";
            if (this.trainingData.conversations.length > 0) {
                response = this.trainingData.conversations[
                    responseIndex % this.trainingData.conversations.length
                ]?.output || response;
            }

            // Cache the response
            this.responseCache.set(cacheKey, {
                response,
                timestamp: new Date().getTime()
            });

            // Cleanup
            inputTensor.dispose();
            prediction.dispose();

            return response;

        } catch (error) {
            console.error(chalk.red("❌ Model response error:"), error);
            return null;
        }
    }

    async generateEnhancedResponse(inputText) {
        if (!inputText) return null;

        const possibilities = [];
        const context = this.buildResponseContext(inputText);
        
        const directMatch = this.findClosestMatch(inputText, context);
        if (directMatch) {
            possibilities.push({
                response: this.properlyCapitalize(directMatch.output),
                confidence: directMatch.confidence,
                source: 'direct_match'
            });
        }

        const { bestMatch, score } = this.matcher.findBestTemplate(inputText);
        if (bestMatch && score < 3) {
            possibilities.push({
                response: this.properlyCapitalize(bestMatch.output),
                confidence: 1 - (score / 10),
                source: 'template'
            });
        }

        const similarResponses = this.findSimilarResponses(inputText, context);
        possibilities.push(...similarResponses.map(resp => ({
            response: this.properlyCapitalize(resp.output),
            confidence: resp.similarity,
            source: 'historical'
        })));

        const modelResponse = await this.generateModelResponse(inputText);
        if (modelResponse) {
            possibilities.push({
                response: this.properlyCapitalize(modelResponse),
                confidence: 0.6,
                source: 'ai_model'
            });
        }

        // Learn from this interaction in the background
        this.learnFromInteraction(inputText, possibilities[0]?.response || "").catch(console.error);

        return possibilities
            .sort((a, b) => b.confidence - a.confidence)
            .slice(0, 3);
    }

    async learnFromInteraction(input, output) {
        try {
            const inputTokens = this.tokenizer.tokenize(input.toLowerCase());
            const outputTokens = this.tokenizer.tokenize(output.toLowerCase());

            // Prepare training data
            const paddedInput = [...inputTokens.slice(0, 50), ...Array(Math.max(0, 50 - inputTokens.length)).fill(0)];
            const paddedOutput = [...outputTokens.slice(0, 50), ...Array(Math.max(0, 50 - outputTokens.length)).fill(0)];

            // Convert to tensors
            const xs = tf.tensor2d([paddedInput], [1, 50]);
            const ys = tf.tensor2d([paddedOutput], [1, 50]);

            // Train for one step
            await this.model.trainOnBatch(xs, ys);

            // Cleanup
            xs.dispose();
            ys.dispose();

            // Update cache
            globalCache.lastUpdate = '2025-02-05 04:41:24';

            console.log(chalk.green("✅ Learned from interaction"));
        } catch (error) {
            console.error(chalk.red("❌ Learning error:"), error);
        }
    }

    buildResponseContext(inputText) {
        return {
            recentConversations: this.trainingData.conversations.slice(-5),
            currentTime: this.currentDateTime,
            currentUser: this.currentUser,
            vocabulary: this.vocab
        };
    }

    findClosestMatch(inputText, context) {
        if (!inputText || !this.trainingData.conversations) return null;

        const normalizedInput = inputText.toLowerCase().trim();
        
        let bestMatch = null;
        let bestScore = Infinity;

        this.trainingData.conversations.forEach(conv => {
            if (!conv || !conv.input) return;

            const score = levenshtein.get(normalizedInput, conv.input.toLowerCase().trim());
            if (score < bestScore && score < normalizedInput.length * 0.4) {
                bestScore = score;
                bestMatch = {
                    output: conv.output,
                    confidence: 1 - (score / Math.max(normalizedInput.length, conv.input.length))
                };
            }
        });

        return bestMatch;
    }

    findSimilarResponses(inputText, context) {
        if (!inputText || !this.trainingData.conversations) return [];

        const inputTokens = this.tokenizer.tokenize(inputText.toLowerCase());
        
        return this.trainingData.conversations
            .map(conv => {
                if (!conv || !conv.input || !conv.output) return null;

                const convTokens = this.tokenizer.tokenize(conv.input.toLowerCase());
                const commonTokens = inputTokens.filter(token => 
                    convTokens.includes(token)
                );

                const tfidfSimilarity = this.calculateTFIDFSimilarity(inputText, conv.input);
                const tokenSimilarity = commonTokens.length / 
                    Math.max(inputTokens.length, convTokens.length);

                const similarity = (tfidfSimilarity * 0.7) + (tokenSimilarity * 0.3);

                return {
                    ...conv,
                    similarity
                };
            })
            .filter(conv => conv && conv.similarity > 0.3)
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, 5);
    }

    calculateTFIDFSimilarity(text1, text2) {
        if (!text1 || !text2) return 0;

        const tfidf = new natural.TfIdf();
        tfidf.addDocument(text1.toLowerCase());
        tfidf.addDocument(text2.toLowerCase());

        let similarity = 0;
        const terms = new Set([
            ...this.tokenizer.tokenize(text1.toLowerCase()),
            ...this.tokenizer.tokenize(text2.toLowerCase())
        ]);

        terms.forEach(term => {
            const score1 = tfidf.tfidf(term, 0);
            const score2 = tfidf.tfidf(term, 1);
            similarity += Math.min(score1, score2);
        });

        return similarity / terms.size;
    }

    cleanText(text) {
        if (!text) return '';
        
        return text
            .trim()
            .replace(/\s+/g, ' ')
            .replace(/[^\w\s\-',.!?]/g, '')
            .replace(/\s+([,.!?])/g, '$1');
    }
}

module.exports = ResponseGenerator;
