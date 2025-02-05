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
        this.loadTrainingData();
        this.setupModelCache();
    }

    setupModelCache() {
        // Setup LRU cache for model outputs
        this.modelCache.maxSize = 2500;
        this.responseCache.maxSize = 500;

        // Clean old cache entries periodically
        setInterval(() => {
            const now = new Date().getTime();
            for (const [key, value] of this.modelCache) {
                if (now - value.timestamp > 3600000) { // 1 hour period
                    this.modelCache.delete(key);
                }
            }
        }, 900000); // Clean every 15 minutes for stability on RAM to prevent overflow
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
            optimizer: tf.train.adam(1.85),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        console.log(chalk.green("✅ New model created"));
        return model;

    } catch (error) {
        console.log(chalk.red("❌ Training data not found"), Error);
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
                console.error(`❌ Model was not able to load training data from ${this.trainingDataPath}`);
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
  
    async updateTrainingData(input, output) {
    this.trainingData.conversations.push({
        input: input,
        output: output,
        timestamp: new Date().toISOString(),
        user: this.currentUser
    });
      await this.saveTrainingData(); // Save updated training data
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

        await this.trainingData(
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
            console.log(chalk.yellow('⚠ No matches found ⚠'));
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
        // Prepare input - ensure we have a valid vocabulary mapping
        const tokens = this.tokenizer.tokenize(inputText.toLowerCase());
        
        // Map tokens to vocabulary indices, using 0 for unknown tokens
        const tokenIndices = tokens.map(token => this.vocab[token] || 0);
        
        // Pad or truncate to fixed length (50)
        const paddedTokens = [...tokenIndices.slice(0, 50), ...Array(Math.max(0, 50 - tokenIndices.length)).fill(0)];

        // Convert to tensor with proper shape
        const inputTensor = tf.tensor2d([paddedTokens], [1, 50]);
        
        // Get prediction
        const prediction = this.model.predict(inputTensor);
        
        // Get response from training data
        let response;
        if (prediction.shape[1] === this.trainingData.conversations.length) {
            const responseIndex = tf.argMax(prediction, 1).dataSync()[0];
            response = this.trainingData.conversations[responseIndex]?.output;
        }
        
        if (!response) {
            // Fallback to closest matching response
            const { bestMatch } = this.matcher.findBestTemplate(inputText);
            response = bestMatch?.output || "I'm still learning how to respond to that.";
        }

        // Cache the response
        this.responseCache.set(cacheKey, {
            response,
            timestamp: new Date().getTime()
        });

        // Cleanup tensors
        inputTensor.dispose();
        prediction.dispose();

        return response;

    } catch (error) {
        console.error(chalk.red("❌ Model response error:"), error);
        return "I encountered an error while processing your message.";
    }
}

async learnFromInteraction(input, output) {
    try {
        if (!input || !output) return;

        // Convert tokens to numeric indices
        const inputTokens = this.tokenizer.tokenize(input.toLowerCase());
        const inputIndices = inputTokens.map(token => {
            if (!this.vocab[token]) {
                this.vocab[token] = Object.keys(this.vocab).length + 1;
            }
            return parseInt(this.vocab[token]); // Ensure numeric
        });

        // Create padded numeric input sequence
        const paddedInput = [...inputIndices.slice(0, 50), ...Array(Math.max(0, 50 - inputIndices.length)).fill(0)];

        // Convert output tokens to numeric indices
        const outputTokens = this.tokenizer.tokenize(output.toLowerCase());
        const outputIndices = outputTokens.map(token => {
            if (!this.vocab[token]) {
                this.vocab[token] = Object.keys(this.vocab).length + 1;
            }
            return parseInt(this.vocab[token]); // Ensure numeric
        });

        // Create padded numeric output sequence
        const paddedOutput = [...outputIndices.slice(0, 16), ...Array(Math.max(0, 16 - outputIndices.length)).fill(0)];

        // Create tensors with numeric values
        const inputTensor = tf.tensor2d([paddedInput], [1, 50]);
        
        // Create one-hot encoded output
        const oneHotOutput = tf.oneHot(tf.tensor1d(paddedOutput, 'int32'), Math.max(...Object.values(this.vocab)) + 1);

        // Train for one step
        await this.model.trainOnBatch(inputTensor, oneHotOutput);

        // Cleanup tensors
        inputTensor.dispose();
        oneHotOutput.dispose();

        // Update training data
        await this.updateTrainingData(input, output);

        // Update cache
        globalCache.lastUpdate = this.currentDateTime;

        console.log(chalk.green("✅ Learned from interaction"));
    } catch (error) {
        console.error(chalk.red("❌ Learning error:"), error);
        // Log additional debug info
        console.log("Vocab:", this.vocab);
        console.log("Input:", input);
        console.log("Output:", output);
    }
}

    async generateEnhancedResponse(inputText, chatHistory = []) {
        if (!inputText) return null;

        const possibilities = [];
        const context = this.buildResponseContext(inputText, chatHistory);
        
        // Get contextual understanding
        const understanding = await this.analyzeContext(inputText, chatHistory);
        
        // Check for conversation continuity
        const conversationThread = this.findConversationThread(inputText, chatHistory);
        
        const directMatch = this.findClosestMatch(inputText, context);
        if (directMatch) {
            possibilities.push({
                response: this.properlyCapitalize(this.addContextToResponse(directMatch.output, understanding)),
                confidence: directMatch.confidence * (conversationThread ? 1.2 : 1),
                source: 'direct_match'
            });
        }

        // Get similar responses considering context
        const similarResponses = await this.findSimilarResponsesWithContext(inputText, context, understanding);
        possibilities.push(...similarResponses);

        // Generate model response with context
        const modelResponse = await this.generateModelResponseWithContext(inputText, understanding, chatHistory);
        if (modelResponse) {
            possibilities.push({
                response: this.properlyCapitalize(modelResponse),
                confidence: 0.6 * (conversationThread ? 1.2 : 1),
                source: 'ai_model'
            });
        }

        // Learn from this interaction
        await this.learnFromInteraction(inputText, possibilities[0]?.response || "", understanding);

        return possibilities
            .sort((a, b) => b.confidence - a.confidence)
            .slice(0, 3);
    }

    async analyzeContext(input, chatHistory) {
        const understanding = {
            topic: await this.detectTopic(input),
            references: await this.findReferences(input),
            sentiment: this.analyzeSentiment(input),
            previousContext: this.extractPreviousContext(chatHistory),
            userIntent: this.detectUserIntent(input, chatHistory)
        };

        return understanding;
    }

    async detectTopic(input) {
        // Use TF-IDF to find key terms
        const keyTerms = this.extractKeyTerms(input);
        
        // Try to find related topics in knowledge base
        const relatedTopics = await Promise.all(
            keyTerms.map(term => this.searchKnowledgeBase(term))
        );

        return {
            mainTopic: keyTerms[0],
            relatedTopics: relatedTopics.filter(Boolean)
        };
    }

    async findReferences(input) {
        if (!input) return [];

        try {
            const references = [];
            const keyTerms = this.extractKeyTerms(input);

            // Search for references in training data
            for (const term of keyTerms) {
                const matches = this.trainingData.conversations.filter(conv => 
                    conv.input.toLowerCase().includes(term.toLowerCase()) ||
                    conv.output.toLowerCase().includes(term.toLowerCase())
                );

                matches.forEach(match => {
                    if (match.source) {
                        references.push({
                            term,
                            source: match.source,
                            confidence: this.calculateSimilarity(input, match.input)
                        });
                    }
                });
            }

            // Search Wikipedia if enabled and no references found
            if (references.length === 0) {
                for (const term of keyTerms.slice(0, 2)) { // Limit to first 2 terms
                    const wikiInfo = await this.searchWikipedia(term);
                    if (wikiInfo) {
                        references.push({
                            term,
                            source: 'Wikipedia',
                            confidence: 0.7
                        });
                    }
                }
            }

            // Sort by confidence and remove duplicates
            return Array.from(new Set(references
                .sort((a, b) => b.confidence - a.confidence)
                .map(ref => ref.source)
            ));

        } catch (error) {
            console.error("Error finding references:", error);
            return [];
        }
    }

    calculateSimilarity(str1, str2) {
        if (!str1 || !str2) return 0;
        
        const set1 = new Set(str1.toLowerCase().split(' '));
        const set2 = new Set(str2.toLowerCase().split(' '));
        
        const intersection = new Set([...set1].filter(x => set2.has(x)));
        const union = new Set([...set1, ...set2]);
        
        return intersection.size / union.size;
    }

    findConversationThread(input, history) {
        if (!history || history.length === 0) return null;

        // Look for conversation continuity markers
        const continuityMarkers = [
            'that', 'it', 'this', 'those', 'these', 'they',
            'the', 'your', 'my', 'our', 'their'
        ];

        const hasMarkers = continuityMarkers.some(marker => 
            input.toLowerCase().includes(marker)
        );

        if (hasMarkers) {
            // Find the most recent relevant message
            const relevantMessage = history
                .slice()
                .reverse()
                .find(msg => {
                    const similarity = this.calculateSimilarity(input, msg.text);
                    return similarity > 0.3;
                });

            if (relevantMessage) {
                return {
                    previousMessage: relevantMessage,
                    continuityScore: this.calculateContinuityScore(input, relevantMessage)
                };
            }
        }

        return null;
    }

    addContextToResponse(response, understanding) {
        if (!understanding || !response) return response;

        // Add contextual references if needed
        if (understanding.previousContext) {
            response = this.addPreviousContext(response, understanding.previousContext);
        }

        // Add source citations if available
        if (understanding.references && understanding.references.length > 0) {
            response += '\n\nSources: ' + understanding.references.join(', ');
        }

        return response;
    }

    async generateModelResponseWithContext(input, understanding, history) {
        // Prepare context-aware input
        const contextInput = this.prepareContextInput(input, understanding, history);
        
        // Generate response using the enhanced input
        const response = await this.generateModelResponse(contextInput);
        
        // Post-process response with context
        return this.addContextToResponse(response, understanding);
    }

    prepareContextInput(input, understanding, history) {
        let contextInput = input;

        // Add recent relevant history
        if (history && history.length > 0) {
            const relevantHistory = history
                .slice(-3)
                .map(msg => `${msg.sender}: ${msg.text}`)
                .join('\n');
            contextInput = `Previous messages:\n${relevantHistory}\n\nCurrent message: ${input}`;
        }

        // Add topic context if available
        if (understanding.topic) {
            contextInput += `\nContext: ${understanding.topic.mainTopic}`;
        }

        return contextInput;
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

    async findSimilarResponsesWithContext(inputText, context, understanding) {
        if (!inputText || !this.trainingData.conversations) return [];

        const responses = [];
        const topicKeywords = understanding.topic?.mainTopic ? [understanding.topic.mainTopic] : [];
        const sentimentScore = understanding.sentiment?.score || 0;

        // Get base similarity matches
        const baseMatches = this.findSimilarResponses(inputText, context);

        for (const match of baseMatches) {
            let contextualConfidence = match.similarity;

            // Boost confidence if topics match
            if (topicKeywords.length > 0) {
                const matchTopics = this.extractKeyTerms(match.input);
                const topicOverlap = topicKeywords.filter(topic => 
                    matchTopics.includes(topic)
                ).length;
                contextualConfidence *= (1 + (topicOverlap * 0.2));
            }

            // Adjust for sentiment alignment
            const matchSentiment = this.analyzeSentiment(match.output).score;
            const sentimentAlignment = 1 - (Math.abs(sentimentScore - matchSentiment) / 2);
            contextualConfidence *= sentimentAlignment;

            responses.push({
                response: this.properlyCapitalize(match.output),
                confidence: contextualConfidence,
                source: 'contextual_match'
            });
        }

        // Sort by confidence and return top matches
        return responses
            .sort((a, b) => b.confidence - a.confidence)
            .slice(0, 3);
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

    extractKeyTerms(text) {
        if (!text) return [];

        // Remove punctuation and convert to lowercase
        const cleanText = text.toLowerCase().replace(/[^\w\s]/g, '');
        
        // Tokenize
        const tokens = this.tokenizer.tokenize(cleanText);
        
        // Remove stopwords
        const stopwords = new Set([
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 
            'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 
            'that', 'the', 'to', 'was', 'were', 'will', 'with'
        ]);
        
        const filteredTokens = tokens.filter(token => !stopwords.has(token));
        
        // Calculate term frequency
        const termFreq = {};
        filteredTokens.forEach(token => {
            termFreq[token] = (termFreq[token] || 0) + 1;
        });
        
        // Sort by frequency
        const sortedTerms = Object.entries(termFreq)
            .sort(([,a], [,b]) => b - a)
            .map(([term]) => term);
        
        return sortedTerms.slice(0, 5); // Return top 5 terms
    }

    async searchKnowledgeBase(term) {
        if (!term) return null;

        try {
            // Check cache first
            const cacheKey = `kb_${term.toLowerCase()}`;
            if (this.modelCache.has(cacheKey)) {
                return this.modelCache.get(cacheKey).data;
            }

            // Search training data first
            const relevantData = this.trainingData.conversations.find(conv => 
                conv.input.toLowerCase().includes(term.toLowerCase()) ||
                conv.output.toLowerCase().includes(term.toLowerCase())
            );

            if (relevantData) {
                this.modelCache.set(cacheKey, {
                    data: relevantData.output,
                    timestamp: Date.now()
                });
                return relevantData.output;
            }

            // If not found in training data, try Wikipedia
            const wikiResult = await this.searchWikipedia(term);
            if (wikiResult) {
                this.modelCache.set(cacheKey, {
                    data: wikiResult,
                    timestamp: Date.now()
                });
                return wikiResult;
            }

            return null;
        } catch (error) {
            console.error(`Error searching knowledge base for term "${term}":`, error);
            return null;
        }
    }

    async searchWikipedia(term) {
        try {
            const wiki = require('wikijs').default;
            const searchResults = await wiki().search(term);
            if (searchResults.results && searchResults.results.length > 0) {
                const page = await wiki().page(searchResults.results[0]);
                const summary = await page.summary();
                return summary;
            }
            return null;
        } catch (error) {
            console.error(`Error searching Wikipedia for term "${term}":`, error);
            return null;
        }
    }

    analyzeSentiment(text) {
        if (!text) return { score: 0, label: 'neutral' };

        // Simple rule-based sentiment analysis
        const positiveWords = new Set([
            'good', 'great', 'awesome', 'excellent', 'happy', 
            'love', 'wonderful', 'fantastic', 'amazing', 'thanks'
        ]);
        
        const negativeWords = new Set([
            'bad', 'terrible', 'awful', 'horrible', 'sad', 
            'hate', 'poor', 'worst', 'annoying', 'sorry'
        ]);

        const words = text.toLowerCase().split(/\s+/);
        let score = 0;

        words.forEach(word => {
            if (positiveWords.has(word)) score += 1;
            if (negativeWords.has(word)) score -= 1;
        });

        return {
            score,
            label: score > 0 ? 'positive' : score < 0 ? 'negative' : 'neutral'
        };
    }

    extractPreviousContext(chatHistory) {
        if (!chatHistory || chatHistory.length === 0) return null;

        // Get last 3 messages for context
        const recentMessages = chatHistory.slice(-3);
        
        return {
            lastMessage: recentMessages[recentMessages.length - 1],
            recentContext: recentMessages.map(msg => ({
                role: msg.sender.toLowerCase(),
                content: msg.text
            })),
            topics: this.extractKeyTerms(
                recentMessages.map(msg => msg.text).join(' ')
            )
        };
    }

    detectUserIntent(input, chatHistory) {
        const intents = {
            QUESTION: /^(what|who|where|when|why|how|can|could|would|will|do|does|did|is|are|was|were)/i,
            GREETING: /^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))/i,
            FAREWELL: /^(bye|goodbye|see\s+you|farewell)/i,
            GRATITUDE: /(thank|thanks)/i,
            REQUEST: /^(please|can\s+you|could\s+you|would\s+you)/i,
            CONFIRMATION: /^(yes|yeah|yep|sure|okay|ok|alright)/i,
            NEGATION: /^(no|nope|nah|not)/i
        };

        for (const [intent, pattern] of Object.entries(intents)) {
            if (pattern.test(input.trim())) {
                return {
                    type: intent,
                    confidence: 0.8,
                    metadata: {
                        pattern: pattern.source,
                        match: input.match(pattern)[0]
                    }
                };
            }
        }

        // Try to infer intent from context if no pattern matches
        if (chatHistory && chatHistory.length > 0) {
            const lastMessage = chatHistory[chatHistory.length - 1];
            if (lastMessage.sender === 'AI' && lastMessage.text.endsWith('?')) {
                return {
                    type: 'RESPONSE_TO_QUESTION',
                    confidence: 0.6,
                    metadata: {
                        previousQuestion: lastMessage.text
                    }
                };
            }
        }

        return {
            type: 'UNKNOWN',
            confidence: 0.3,
            metadata: {}
        };
    }

    calculateContinuityScore(input, previousMessage) {
        if (!input || !previousMessage || !previousMessage.text) return 0;

        const baseSimilarity = this.calculateSimilarity(input, previousMessage.text);
        const timeDecay = previousMessage.timestamp ? 
            Math.exp(-(Date.now() - new Date(previousMessage.timestamp).getTime()) / (1000 * 60 * 60)) : 1;

        return baseSimilarity * timeDecay;
    }

    addPreviousContext(response, context) {
        if (!context || !context.lastMessage) return response;

        const contextReferences = {
            'it': context.lastMessage.text,
            'that': context.lastMessage.text,
            'this': context.lastMessage.text
        };

        // Replace pronouns with their context
        Object.entries(contextReferences).forEach(([pronoun, reference]) => {
            const regex = new RegExp(`\\b${pronoun}\\b`, 'gi');
            if (response.match(regex)) {
                response = response.replace(regex, `"${reference}"`);
            }
        });

        return response;
    }

    // Replaces the existing trainTransformerModel method with this enhanced version:
    async trainTransformerModel(model, data, labels, maxEpochs = 10, batchSize = 4) {
        console.log("⏳ Starting staged training process... ⏳");
        
        const reshapedData = data.map(seq => seq.map(step => [step]));
        const xs = tf.tensor3d(reshapedData, [reshapedData.length, reshapedData[0].length, 1]);
        const ys = tf.tensor2d(labels, [labels.length, labels[0].length]);
        const dataset = tf.data.zip({ xs: tf.data.array(xs), ys: tf.data.array(ys) }).batch(batchSize);
    
        let currentEpoch = 0;
        const TIMEOUT_PER_EPOCH = 30000; // 30 seconds in timeout timer
        
        // Train for the specified number of epochs & log the current stats on console/terminal
        while (currentEpoch < maxEpochs) {
            console.log(chalk.green(`\n📊 Starting Epoch ${currentEpoch + 1}/${maxEpochs} ⏳`));
            
            try {
                // Create a promise that either resolves with training or rejects after timeout
                await Promise.race([
                    // Training promise
                    model.fitDataset(dataset, {
                        epochs: 1,
                        callbacks: {
                            onBatchEnd: (batch, logs) => {
                                console.log(`  ▸ Batch ${batch}: loss = ${logs.loss.toFixed(4)}`);
                            },
                            onEpochEnd: (epoch, logs) => {
                                console.log(`✅ Epoch ${currentEpoch + 1} completed - Loss: ${logs.loss.toFixed(4)}`);
                            }
                        }
                    }),
                    
                    // Timeout promise
                    new Promise((_, reject) => {
                        setTimeout(() => {
                            reject(new Error('Epoch timeout ⌛'));
                        }, TIMEOUT_PER_EPOCH);
                    })
                ]);
    
            } catch (error) {
                if (error.message === 'Epoch timeout') {
                    console.warn(`⚠️ Epoch ${currentEpoch + 1} timed out after ${TIMEOUT_PER_EPOCH/1000}s`);
                } else {
                    console.error(`❌ Error in epoch ${currentEpoch + 1}:`, error);
                }
            }
    
            // Small delay between epochs to prevent system overload
            await new Promise(resolve => setTimeout(resolve, 1000));
            currentEpoch++;

            
            // Save intermediate model state every 3 epochs (Checkpoint State)
            if (currentEpoch % 3 === 0) {
                try {
                    await this.saveIntermediateModel(currentEpoch);
                    console.log(`💾 Saved intermediate model at epoch ${currentEpoch}`);
                } catch (error) {
                    console.warn(`⚠️ Failed to save intermediate model:`, error);
                }
            }
        }
    
        // Cleanup tensors
        xs.dispose();
        ys.dispose();
    
        console.log("\n🏁 Training completed! ✅");
        return model;
    }
    
    // Helper method to the class:
    async saveIntermediateModel(epoch) {
        const savePath = `${this.modelPath}/intermediate_epoch_${epoch}`;
        await this.model.save(`file://${savePath}`);
    }
    async saveTrainingData(epoch) {
        const savePath = `${this.modelPath}/intermediate_epoch_${epoch}`;
        await this.model.save(`file://${savePath}`);
    }

    async findOrCreateDefinition(word) {
        if (!word) return null;
    
        // Check existing definitions
        const existingDef = this.trainingData.vocabulary.definitions?.find(
            def => def.word.toLowerCase() === word.toLowerCase()
        );
    
        if (existingDef) {
            return existingDef.definition;
        }
    
        // If no definition exists, search Wikipedia
        try {
            const wiki = require('wikijs').default;
            const searchResults = await wiki().search(word);
            if (searchResults.results && searchResults.results.length > 0) {
                const page = await wiki().page(searchResults.results[0]);
                const summary = await page.summary();
                
                // Extract first sentence as definition
                const definition = summary.split(/[.!?](?:\s|$)/)[0] + '.';
                
                // Add to training data
                if (!this.trainingData.vocabulary.definitions) {
                    this.trainingData.vocabulary.definitions = [];
                }
                
                this.trainingData.vocabulary.definitions.push({
                    word: word,
                    definition: definition
                });
                
                // Save training data
                this.saveTrainingData();
                
                return definition;
            }
        } catch (error) {
            console.error(`❌ Error getting definition for "${word}":`, error);
        }
        
        return null;
    }
}

module.exports = ResponseGenerator;
