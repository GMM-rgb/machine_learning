const express = require('express');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');
const axios = require('axios');
const { app, BrowserWindow } = require('electron');
const wiki = require('wikijs').default;
const math = require('mathjs');
const { ifError } = require('assert');
const tf = require('@tensorflow/tfjs-node'); // Correct import of TensorFlow.js
const https = require('https');

const expressApp = express();
const PORT = 3000;

let chatEnabled = true; // Flag to control chat availability

// Middleware to parse JSON and serve static files
expressApp.use(bodyParser.json());
expressApp.use(express.static(path.join(__dirname, 'public')));

// Normalize input by replacing contractions and short term meaning
function normalizeText(input) {
    const maps = {
        contractions: {
            "its": "it's",
            "im": "i'm",
            "youre": "you're",
            "theyre": "they're",
            "were": "we're", 
            "hes": "he's",
            "shes": "she's",
            "thats": "that is",
            "cant": "cannot",
            "dont": "do not",
            "doesnt": "does not",
            "wont": "will not",
            "isnt": "is not",
            "arent": "are not",
            "werent": "were not",
            "hasnt": "has not",
            "havent": "have not",
            "didnt": "did not",
            "wouldnt": "would not",
            "couldnt": "could not",
            "shouldnt": "should not",
            "mightnt": "might not",
        },
        slang: {
            "idk": "I don't know",
            "idr": "I don't remember",
            "omg": "Oh my God",
            "btw": "By the way",
            "lol": "Laugh out loud",
            "brb": "Be right back",
            "gtg": "Got to go",
            "ttyl": "Talk to you later",
            "fyi": "For your information",
            "smh": "Shaking my head",
            "lmao": "Laughing my ass off",
            "bff": "Best friends forever",
            "tbh": "To be honest",
            "yolo": "You only live once",
            "nvm": "Never mind",
        }
    };

    const words = input.split(/\s+/);
    return words
        .map(word => maps.contractions[word.toLowerCase()] || maps.slang[word.toLowerCase()] || word)
        .join(' ');
}

// Load or initialize knowledge
const knowledgeFile = 'knowledge.json';
let knowledge = {};

if (fs.existsSync(knowledgeFile)) {
    knowledge = JSON.parse(fs.readFileSync(knowledgeFile, 'utf8'));
} else {
    fs.writeFileSync(knowledgeFile, JSON.stringify(knowledge, null, 2));
}

// Initialize training data storage
const trainingDataFile = 'training_data.json';
let trainingData = {
    conversations: [],
    vocabulary: {},
    lastTrainingDate: null
};

// Load or create training data file
if (fs.existsSync(trainingDataFile)) {
    trainingData = JSON.parse(fs.readFileSync(trainingDataFile, 'utf8'));
} else {
    fs.writeFileSync(trainingDataFile, JSON.stringify(trainingData, null, 2));
}

// Function to save training data
function saveTrainingData() {
    trainingData.lastTrainingDate = new Date().toISOString();
    fs.writeFileSync(trainingDataFile, JSON.stringify(trainingData, null, 2));
}

// Function to add new training data
function addTrainingData(userMessage, aiResponse) {
    trainingData.conversations.push({
        input: userMessage,
        output: aiResponse,
        timestamp: new Date().toISOString()
    });

    // Update vocabulary
    const words = preprocessText(userMessage + ' ' + aiResponse);
    words.forEach(word => {
        if (!trainingData.vocabulary[word]) {
            trainingData.vocabulary[word] = Object.keys(trainingData.vocabulary).length + 1;
        }
    });

    saveTrainingData();

    // Retrain model if we have enough new data (every 5 conversations)
    if (trainingData.conversations.length % 5 === 0) {
        retrainModel();
    }
}

// Function to retrain the model with accumulated data
async function retrainModel() {
    const data = [];
    const labels = [];

    // Process all conversations for training
    trainingData.conversations.forEach(conversation => {
        const inputWords = preprocessText(conversation.input);
        const outputWords = preprocessText(conversation.output);

        // Create training pairs
        inputWords.forEach((word, index) => {
            if (!vocab[word]) {
                vocab[word] = Object.keys(vocab).length + 1;
            }
            if (index < inputWords.length - 1) {
                data.push(inputWords.slice(0, index + 1).map(w => vocab[w]));
                labels.push([vocab[inputWords[index + 1]]]);
            }
        });

        // Also learn from AI responses
        outputWords.forEach((word, index) => {
            if (!vocab[word]) {
                vocab[word] = Object.keys(vocab).length + 1;
            }
            if (index < outputWords.length - 1) {
                data.push(outputWords.slice(0, index + 1).map(w => vocab[w]));
                labels.push([vocab[outputWords[index + 1]]]);
            }
        });
    });

    // Retrain the model
    model = createModel(Object.keys(vocab).length);
    await trainModel(model, data, labels);
    console.log('Model retrained with accumulated data');
}

// Levenshtein distance for closest match
function getLevenshteinDistance(a, b) {
    const tmp = [];
    for (let i = 0; i <= b.length; i++) {
        tmp[i] = [i];
    }
    for (let i = 0; i <= a.length; i++) {
        tmp[0][i] = i;
    }

    for (let i = 1; i <= b.length; i++) {
        for (let j = 1; j <= a.length; j++) {
            tmp[i][j] = b[i - 1] === a[j - 1] ? tmp[i - 1][j - 1] : Math.min(tmp[i - 1][j - 1] + 1, tmp[i][j - 1] + 1, tmp[i - 1][j] + 1);
        }
    }
    return tmp[b.length][a.length];
}

// Find the best match in knowledge (with exact and fuzzy match)
function findBestMatch(query, knowledge) {
    let closestMatch = null;
    let minDistance = Infinity;

    // Check for exact matches first
    for (const key in knowledge) {
        const normalizedKey = key.toLowerCase();
        if (query === normalizedKey) {
            return knowledge[key];
        }
    }

    // Use fuzzy matching based on Levenshtein distance
    for (const key in knowledge) {
        const distance = getLevenshteinDistance(query, key);
        if (distance < minDistance) {
            minDistance = distance;
            closestMatch = key;
        }
    }

    return closestMatch ? knowledge[closestMatch] : "Sorry, I didn't understand that.";
}

// Detect if the user is asking for Wikipedia information
function isWikipediaQuery(input) {
    const wikipediaTriggers = [
        /^what is |^what are |^what was |^tell me about |^who is |^explain |^define |^describe /i,
        /^how does |^how do |^how can |^what does |^where is |^when did /i
    ];

    const normalizedInput = input.toLowerCase().trim();
    return wikipediaTriggers.some(trigger => trigger.test(normalizedInput));
}

// Function to summarize text into bullet points
function summarizeText(text) {
    const sentences = text.split('. ');
    const summary = sentences.slice(0, 5).map(sentence => `• ${sentence.trim()}`).join('\n');
    return summary;
}

// Wikipedia info fetching with input sanitization (e.g., "What is X?") 
async function getWikipediaInfo(query, summarize = false) {
    // Clean up query
    const sanitizedQuery = query
        .toLowerCase()
        .replace(/^(what is|what are|what was|tell me about|who is|explain|define|describe|how does|how do|how can|what does|where is|when did)\s+/i, '')
        .replace(/[?.,!]/g, '')
        .trim();

    console.log("Sanitized Wikipedia query:", sanitizedQuery);

    try {
        const searchResults = await wiki().search(sanitizedQuery);
        if (!searchResults.results || !searchResults.results.length) {
            return `Sorry, I couldn't find any relevant information on Wikipedia about ${query}.`;
        }

        const page = await wiki().page(searchResults.results[0]);
        const summary = await page.summary();
        
        if (summarize) {
            return `Here's a summary of what I found on Wikipedia:\n${summarizeText(summary)}`;
        }
        return `Here's what I found on Wikipedia: ${summary}`;
    } catch (error) {
        console.error(`Error fetching data from Wikipedia for query "${sanitizedQuery}":`, error);
        return `Sorry, I couldn't find any relevant information on Wikipedia about ${query}.`;
    }
}

// Update Bing search function with better error handling and logging
async function getBingSearchInfo(query) {
    // You'll need to get a valid API key from Microsoft Azure
    const subscriptionKey = '1feda3372abf425494ce986ad9024238';
    const endpoint = 'https://api.bing.microsoft.com/v7.0/search';

    try {
        chatEnabled = false; // Disable chat while searching
        console.log('Initiating Bing search for:', query);

        const response = await axios({
            method: 'get',
            url: endpoint,
            headers: {
                'Ocp-Apim-Subscription-Key': subscriptionKey,
                'Accept': 'application/json'
            },
            params: {
                q: query,
                count: 1,
                responseFilter: 'Webpages',
                mkt: 'en-US'
            }
        });

        console.log('Bing API response status:', response.status);
        
        if (response.data && response.data.webPages && response.data.webPages.value && response.data.webPages.value.length > 0) {
            const result = response.data.webPages.value[0];
            chatEnabled = true;
            return `Here's what I found on Bing: ${result.name}\n${result.snippet}\nSource: ${result.url}`;
        } else {
            console.log('No results found in Bing response:', response.data);
            return "Sorry, I couldn't find any relevant information on Bing.";
        }
    } catch (error) {
        console.error('Bing search error:', error.response ? error.response.data : error.message);
        chatEnabled = true;
        if (error.response && error.response.status === 401) {
            return "Sorry, there's an issue with the Bing search authentication. Please check the API key.";
        }
        return "Sorry, I couldn't complete the Bing search at this time.";
    } finally {
        chatEnabled = true; // Always re-enable chat
    }
}

// Update handleUserInput function to better handle Bing searches
async function handleUserInput(input) {
    const searchKeywords = ['search bing', 'bing'];
    const lowerCaseInput = input.toLowerCase().trim();

    // Check if input starts with any search keywords
    for (const keyword of searchKeywords) {
        if (lowerCaseInput.startsWith(keyword)) {
            // Extract the actual search query
            const query = lowerCaseInput
                .replace(keyword, '')
                .trim();

            if (!query) {
                return 'Please provide a search query.';
            }

            console.log('Processing Bing search for:', query);
            return await getBingSearchInfo(query);
        }
    }

    return "I'm not sure how to handle that request.";
}

// Function to preprocess text for TensorFlow (AI) model
function preprocessText(text) {
    return text.toLowerCase().replace(/[^a-z0-9\s]/g, '').split(' ');
}

// Function to create a TensorFlow (AI) model
function createModel(vocabSize) {
    const model = tf.sequential();
    model.add(tf.layers.embedding({ inputDim: vocabSize, outputDim: 128 }));
    model.add(tf.layers.lstm({ units: 128, returnSequences: true }));
    model.add(tf.layers.lstm({ units: 128 }));
    model.add(tf.layers.dense({ units: vocabSize, activation: 'softmax' }));
    model.compile({ optimizer: 'adam', loss: 'sparseCategoricalCrossentropy' });
    return model;
}

// Function to train the TensorFlow (AI) model
async function trainModel(model, data, labels, epochs = 10) {
    const xs = tf.tensor2d(data, [data.length, data[0].length]);
    const ys = tf.tensor2d(labels, [labels.length, labels[0].length]);
    await model.fit(xs, ys, { epochs });
}

// Function to predict the next word using the TensorFlow (AI) model
async function predictNextWord(model, inputText, vocab) {
    const input = preprocessText(inputText);
    const inputTensor = tf.tensor2d([input.map(word => vocab[word] || 0)], [1, input.length]);
    const prediction = model.predict(inputTensor);
    const predictedIndex = prediction.argMax(-1).dataSync()[0];
    return Object.keys(vocab).find(key => vocab[key] === predictedIndex);
}

// Initialize vocabulary and model
const vocab = {};
let model;

// Train the model with existing knowledge
async function initializeModel() {
    const data = [];
    const labels = [];

    // Add knowledge base data
    for (const key in knowledge) {
        const words = preprocessText(key);
        words.forEach((word, index) => {
            if (!vocab[word]) {
                vocab[word] = Object.keys(vocab).length + 1;
            }
            if (index < words.length - 1) {
                data.push(words.slice(0, index + 1).map(w => vocab[w]));
                labels.push([vocab[words[index + 1]]]);
            }
        });
    }

    // Add training data
    trainingData.conversations.forEach(conv => {
        const words = preprocessText(conv.input);
        words.forEach((word, index) => {
            if (!vocab[word]) {
                vocab[word] = Object.keys(vocab).length + 1;
            }
            if (index < words.length - 1) {
                data.push(words.slice(0, index + 1).map(w => vocab[w]));
                labels.push([vocab[words[index + 1]]]);
            }
        });
    });

    model = createModel(Object.keys(vocab).length);
    await trainModel(model, data, labels);
}

initializeModel();

// Find similar conversations from training history
function findSimilarConversation(input) {
    const conversations = trainingData.conversations;
    let bestMatch = null;
    let bestScore = 0;

    for (const conv of conversations) {
        const similarity = calculateSimilarity(input.toLowerCase(), conv.input.toLowerCase());
        if (similarity > bestScore && similarity > 0.6) { // Threshold of 60% similarity
            bestScore = similarity;
            bestMatch = conv;
        }
    }

    return bestMatch;
}

// Calculate similarity between two strings
function calculateSimilarity(str1, str2) {
    const words1 = str1.split(' ');
    const words2 = str2.split(' ');
    const commonWords = words1.filter(word => words2.includes(word));
    return commonWords.length / Math.max(words1.length, words2.length);
}

// Get response from training data or knowledge base
function getResponse(message) {
    // First check training data
    const similarConversation = findSimilarConversation(message);
    if (similarConversation) {
        return similarConversation.output;
    }

    // If no match in training data, check knowledge base
    return findBestMatch(message, knowledge);
}

// Update the chat endpoint to better handle Bing searches
expressApp.post('/chat', async (req, res) => {
    if (!chatEnabled) {
        res.json({ response: "Chat is currently disabled while performing a search. Please try again in a moment." });
        return;
    }
    
    const { message } = req.body;
    const normalizedMessage = normalizeText(message.replace(/[,']/g, "").toLowerCase());
    let response = '';

    try {
        // Check for Bing search first (when explicitly requested)
        if (['search bing', 'bing'].some(keyword => normalizedMessage.startsWith(keyword))) {
            console.log('Bing search requested:', message);
            response = await handleUserInput(message);
        }
        // Then check for Wikipedia queries
        else if (isWikipediaQuery(message)) {
            console.log("Wikipedia query detected:", message);
            const summarize = /summarize|summary|bullet points/i.test(message);
            response = await getWikipediaInfo(message, summarize);
            
            // Only fallback to Bing if Wikipedia fails
            if (response.includes("Sorry, I couldn't find any relevant information on Wikipedia")) {
                console.log('Wikipedia failed, trying Bing fallback');
                response = await getBingSearchInfo(message);
            }
        }
        // Finally, try training data and knowledge base
        else {
            response = getResponse(normalizedMessage);
        }

        // Add the interaction to training data if it's not a command
        if (!message.startsWith('/')) {
            addTrainingData(normalizedMessage, response);
        }

        res.json({ response });
    } catch (error) {
        console.error('Error in chat endpoint:', error);
        res.status(500).json({ response: "An error occurred while processing your message." });
    }
});

// Feedback API endpoint
expressApp.post('/feedback', (req, res) => {
    const { message, correctResponse } = req.body;

    const normalizedInput = normalizeText(message.replace(/[,']/g, "").toLowerCase());
    knowledge[normalizedInput] = correctResponse;
    knowledge[message.toLowerCase()] = correctResponse;

    try {
        fs.writeFileSync(knowledgeFile, JSON.stringify(knowledge, null, 2));
        console.log('Knowledge data saved!');
    } catch (err) {
        console.error('Error saving knowledge:', err);
    }

    res.json({ response: "Thank you for your feedback!" });
});

// Try to start local server for AI when /start is posted from HTML JavaScript
const startManualPORT = 53483;
expressApp.post('/start', async (req, res) => {
    // Serve the HTML file
    expressApp.get('/', (req, res) => {
        res.sendFile(path.join(__dirname, 'public', 'AI_HtWebz_Assistant_Version 0.4.html'));
    });
    // Start server
    expressApp.listen(startManualPORT, '0.0.0.0', () => {
        console.log(`Server running on http://localhost:${startManualPORT}`);
    });
});

// Serve the HTML file
expressApp.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'AI_HtWebz_Assistant_Version 0.4.html'));
});

// Start server
expressApp.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://localhost:${PORT}`);
    console.log(`Server startup & setup was successful.`);
});

// Electron App Initialization
let win;

function createWindow() {
    win = new BrowserWindow({
        width: 1250,
        height: 1150,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        },
    });

    win.loadURL(`http://localhost:${PORT}`);
    win.on('closed', () => {
        win = null;
    });
}

app.whenReady().then(() => {
    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
            console.log('BrowserWindow created successfully.');
        }
    }).catch((error) => {
        console.error('Error creating BrowserWindow:', error);
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
  }
});
console.log('Server.js loaded successfully, and has been initialized.');
