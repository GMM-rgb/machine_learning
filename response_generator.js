const TemplateMatcher = require('./template_matcher');
const tf = require('@tensorflow/tfjs-node-gpu');
const fs = require('fs');
const axios = require('axios');
const levenshtein = require('fast-levenshtein');
const { ifError } = require('assert');
const path = require('path');
const { isArray } = require('mathjs');
const readline = require('readline');
const chalk = require('chalk');
const natural = require('natural');
const tokenizer = new natural.WordTokenizer();

// TensorFlow setup - Ensures this is only done once
if (!global.tfSetup) {
    global.tfSetup = true;
}

const model = {};

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
            lastTrainingDate: '2025-02-05 03:56:37'
        };
        this.currentUser = 'GMM-rgb';
        this.currentDateTime = '2025-02-05 03:56:37';
        this.loadModel();
        this.loadTrainingData();
        
        // Initialize Natural Language Processing tools
        this.tokenizer = new natural.WordTokenizer();
        this.sentenceTokenizer = new natural.SentenceTokenizer();
        this.tfidf = new natural.TfIdf();
    }

    async loadModel() {
        try {
            this.model = await tf.loadLayersModel("file://D:/machine_learning/model.json");
            console.log(chalk.green("✅ Model loaded successfully in ResponseGenerator"));
            if(ifError) {
                console.warn(chalk.yellow(`Model not found, creating file for data...`));
                model.createModel(toString());
                this.model.parse(model);
                let model = toString();
                return model;
            }
        } catch (error) {
            console.error(chalk.red("❌ Error loading model in ResponseGenerator:"), error);
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
                
                // Initialize TF-IDF with existing conversations
                this.initializeTFIDF();
                
                console.log(chalk.green("✅ Training data loaded successfully"));
            } catch (error) {
                console.error(chalk.red("❌ Error loading training data:"), error);
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
        console.log(chalk.yellow("⚠️ Initialized empty training data"));
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

        console.log(chalk.cyan('\n=== Training Data Management Console ==='));
        console.log(chalk.yellow('Available Commands:'));
        console.log('1. view    - View training data');
        console.log('2. add     - Add new training data');
        console.log('3. edit    - Edit existing training data');
        console.log('4. delete  - Delete training data');
        console.log('5. search  - Search training data');
        console.log('6. test    - Test response generation');
        console.log('7. stats   - View statistics');
        console.log('8. export  - Export training data');
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
            console.log(chalk.yellow('No training data available.'));
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
            rl.question(chalk.cyan('Enter input text: '), resolve);
        });

        const output = await new Promise(resolve => {
            rl.question(chalk.cyan('Enter output text: '), resolve);
        });

        const capitalizedInput = this.properlyCapitalize(input);
        const capitalizedOutput = this.properlyCapitalize(output);

        await this.updateTrainingData(capitalizedInput, capitalizedOutput, false);
        console.log(chalk.green('✅ Training data added successfully!'));
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
                rl.question(chalk.cyan('\nEnter new input text (or press Enter to keep current): '), resolve);
            });

            const output = await new Promise(resolve => {
                rl.question(chalk.cyan('Enter new output text (or press Enter to keep current): '), resolve);
            });

            this.trainingData.conversations[idx] = {
                input: input.trim() ? this.properlyCapitalize(input) : currentEntry.input,
                output: output.trim() ? this.properlyCapitalize(output) : currentEntry.output,
                timestamp: new Date().toISOString(),
                user: this.currentUser
            };

            await this.saveTrainingData();
            console.log(chalk.green('✅ Training data updated successfully!'));
        } else {
            console.log(chalk.red('❌ Invalid index!'));
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
                rl.question(chalk.yellow('Are you sure you want to delete this entry? (y/n): '), resolve);
            });

            if (confirm.toLowerCase() === 'y') {
                this.trainingData.conversations.splice(idx, 1);
                await this.saveTrainingData();
                console.log(chalk.green('✅ Training data deleted successfully!'));
            } else {
                console.log(chalk.yellow('Deletion cancelled.'));
            }
        } else {
            console.log(chalk.red('❌ Invalid index!'));
        }
    }

    async searchTrainingDataConsole(rl) {
        const searchTerm = await new Promise(resolve => {
            rl.question(chalk.cyan('Enter search term: '), resolve);
        });

        const results = this.trainingData.conversations.filter(conv => 
            conv.input.toLowerCase().includes(searchTerm.toLowerCase()) ||
            conv.output.toLowerCase().includes(searchTerm.toLowerCase())
        );

        if (results.length > 0) {
            console.log(chalk.green(`\nFound ${results.length} matches:`));
            results.forEach((conv, index) => {
                console.log(chalk.yellow(`\n[${index + 1}]`));
                console.log(chalk.cyan('Input:     ') + this.highlightText(conv.input, searchTerm));
                console.log(chalk.cyan('Output:    ') + this.highlightText(conv.output, searchTerm));
                console.log(chalk.cyan('Timestamp: ') + conv.timestamp);
            });
        } else {
            console.log(chalk.yellow('No matches found.'));
        }
    }

    async testResponseGenerationConsole(rl) {
        const input = await new Promise(resolve => {
            rl.question(chalk.cyan('Enter test input: '), resolve);
        });

        console.log(chalk.yellow('\nGenerating responses...'));
        const possibilities = await this.generateEnhancedResponse(input);

        if (possibilities && possibilities.length > 0) {
            console.log(chalk.green('\nGenerated Responses:'));
            possibilities.forEach((p, index) => {
                console.log(chalk.yellow(`\n[${index + 1}] Confidence: ${(p.confidence * 100).toFixed(2)}%`));
                console.log(chalk.cyan('Response: ') + p.response);
                console.log(chalk.cyan('Source:   ') + p.source);
            });
        } else {
            console.log(chalk.red('No responses generated.'));
        }
    }

    async viewStatistics() {
        console.log(chalk.green('\nTraining Data Statistics:'));
        console.log(chalk.cyan('Total Conversations:  ') + this.trainingData.conversations.length);
        console.log(chalk.cyan('Total Definitions:   ') + this.trainingData.definitions.length);
        console.log(chalk.cyan('Vocabulary Size:     ') + Object.keys(this.trainingData.vocabulary).length);
        console.log(chalk.cyan('Last Training Date:  ') + this.trainingData.lastTrainingDate);

        // Calculate average response length
        const avgResponseLength = this.trainingData.conversations.reduce((acc, conv) => 
            acc + (conv.output ? conv.output.length : 0), 0) / 
            (this.trainingData.conversations.length || 1);

        console.log(chalk.cyan('Avg Response Length: ') + avgResponseLength.toFixed(2) + ' characters');
    }

    async exportTrainingData() {
        const exportPath = `training_data_export_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
        try {
            await fs.promises.writeFile(
                exportPath,
                JSON.stringify(this.trainingData, null, 2)
            );
            console.log(chalk.green(`✅ Training data exported successfully to ${exportPath}`));
        } catch (error) {
            console.error(chalk.red('❌ Error exporting training data:'), error);
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
            
            // Special words that should always be capitalized
            const specialWords = ['i', 'i\'m', 'i\'ll', 'i\'ve', 'i\'d'];
            
            return sentence.split(' ').map((word, index) => {
                // Capitalize first word of sentence or special words
                if (index === 0 || specialWords.includes(word.toLowerCase())) {
                    return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
                }
                return word.toLowerCase();
            }).join(' ');
        }).join(' ');
    }

    async generateEnhancedResponse(inputText) {
        if (!inputText) return null;

        const possibilities = [];
        const context = this.buildResponseContext(inputText);
        
        // Get direct match
        const directMatch = this.findClosestMatch(inputText, context);
        if (directMatch) {
            possibilities.push({
                response: this.properlyCapitalize(directMatch.output),
                confidence: directMatch.confidence,
                source: 'direct_match'
            });
        }

        // Get template match
        const { bestMatch, score } = this.matcher.findBestTemplate(inputText);
        if (bestMatch && score < 3) {
            possibilities.push({
                response: this.properlyCapitalize(bestMatch.output),
                confidence: 1 - (score / 10),
                source: 'template'
            });
        }

        // Get similar responses from history
        const similarResponses = this.findSimilarResponses(inputText, context);
        possibilities.push(...similarResponses.map(resp => ({
response: this.properlyCapitalize(resp.output),
            confidence: resp.similarity,
            source: 'historical'
        })));

        // Add AI model generated response if available
        if (this.model) {
            try {
                const modelResponse = await this.generateModelResponse(inputText);
                if (modelResponse) {
                    possibilities.push({
                        response: this.properlyCapitalize(modelResponse),
                        confidence: 0.6, // Base confidence for model responses
                        source: 'ai_model'
                    });
                }
            } catch (error) {
                console.error(chalk.red("❌ Error generating model response:"), error);
            }
        }

        // Sort by confidence and return top 3
        return possibilities
            .sort((a, b) => b.confidence - a.confidence)
            .slice(0, 3);
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

                // Calculate TF-IDF similarity
                const tfidfSimilarity = this.calculateTFIDFSimilarity(inputText, conv.input);
                
                // Calculate token similarity
                const tokenSimilarity = commonTokens.length / 
                    Math.max(inputTokens.length, convTokens.length);

                // Combine both similarities with weights
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

    async generateResponse(inputText) {
        if (!inputText || typeof inputText !== 'string') {
            return "I'm not sure how to respond to that.";
        }

        try {
            // Learn new information in background
            await this.learnInBackground(inputText);
            
            // Generate enhanced responses
            const possibilities = await this.generateEnhancedResponse(inputText);
            
            if (possibilities && possibilities.length > 0) {
                // Get the best response
                const bestResponse = possibilities[0];
                
                // Store the interaction
                await this.updateTrainingData(inputText, bestResponse.response, false);
                
                // Return the best response
                return bestResponse.response;
            }

            return "I'm not sure how to respond to that.";
        } catch (error) {
            console.error(chalk.red("❌ Error in generateResponse:"), error);
            return "I'm having trouble processing that message.";
        }
    }

    async updateTrainingData(input, response, isRefinement) {
        if (!input || !response || !this.trainingData.conversations) return;

        try {
            const existingIndex = this.trainingData.conversations.findIndex(conv => 
                conv && conv.input && conv.input.toLowerCase() === input.toLowerCase()
            );

            const newEntry = {
                input: this.properlyCapitalize(input),
                output: this.properlyCapitalize(response),
                timestamp: new Date(this.currentDateTime).toISOString(),
                user: this.currentUser
            };

            if (existingIndex === -1) {
                // New conversation
                this.trainingData.conversations.push(newEntry);
                
                // Update TF-IDF
                this.tfidf.addDocument(input.toLowerCase());
            } else if (isRefinement) {
                // Update existing conversation
                this.trainingData.conversations[existingIndex] = newEntry;
            }

            // Update last training date
            this.trainingData.lastTrainingDate = this.currentDateTime;
            await this.saveTrainingData();
            
            console.log(chalk.green('✅ Training data updated successfully'));
        } catch (error) {
            console.error(chalk.red("❌ Error updating training data:"), error);
        }
    }

    async saveTrainingData() {
        try {
            // Update last training date before saving
            this.trainingData.lastTrainingDate = this.currentDateTime;
            
            await fs.promises.writeFile(
                this.trainingDataPath,
                JSON.stringify(this.trainingData, null, 2)
            );
            console.log(chalk.green('✅ Training data saved successfully'));
        } catch (error) {
            console.error(chalk.red('❌ Error saving training data:'), error);
        }
    }

    // Utility function to clean and normalize text
    cleanText(text) {
        if (!text) return '';
        
        return text
            .trim()
            .replace(/\s+/g, ' ') // Replace multiple spaces with single space
            .replace(/[^\w\s\-',.!?]/g, '') // Remove special characters except punctuation
            .replace(/\s+([,.!?])/g, '$1'); // Remove spaces before punctuation
    }
}

// Export the ResponseGenerator class
module.exports = ResponseGenerator;
