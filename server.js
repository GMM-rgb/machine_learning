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

    // Retrain model if we have enough new data (every 10 conversations)
    if (trainingData.conversations.length % 10 === 0) {
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
        /what is|what are|tell me about|who is|explain|define|describe/i,
        /how does|how do|how can|can you/i
    ];

    const normalizedInput = input.toLowerCase();
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
    // Clean up punctuation and prepare query
    const sanitizedQuery = query
        .replace(/[^\w\s]/g, '') // Remove punctuation
        .replace(/\b(what is|what are|tell me about|who is|define|describe|how does|how do|how can|can you)\b/gi, '') // Remove common question patterns
        .trim();

    try {
        const page = await wiki().page(sanitizedQuery);
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

// Bing search info fetching
async function getBingSearchInfo(query) {
    const subscriptionKey = '1feda3372abf425494ce986ad9024238';
    const endpoint = `https://api.bing.microsoft.com/v7.0/search?q=${encodeURIComponent(query)}`;

    try {
        chatEnabled = false; // Disable chat while searching
        console.log(`Searching Bing for: ${query}`); // Log the query being searched
        const response = await axios.get(endpoint, {
            headers: {
                'Ocp-Apim-Subscription-Key': subscriptionKey
            }
        });
        console.log('Bing search response:', response.data); // Log the response from Bing

        const topResult = response.data.webPages.value[0];
        chatEnabled = true; // Re-enable chat after search completes
        return `Here's what I found on Bing: ${topResult.name} - ${topResult.snippet} (${topResult.url})`;
    } catch (error) {
        chatEnabled = true; // Ensure chat is re-enabled in case of error
        console.error(`Error fetching data from Bing for query "${query}":`, error);
        return "Sorry, I couldn't find any relevant information on Bing.";
    }
}

async function handleUserInput(input) {
    const searchKeywords = ['search bing', 'bing'];
    const lowerCaseInput = input.toLowerCase();

    for (const keyword of searchKeywords) {
        if (lowerCaseInput.startsWith(keyword)) {
            const query = lowerCaseInput.replace(keyword, '').trim();
            if (query) {
                const cleanedQuery = query.replace(keyword, '').trim();
                const result = await getBingSearchInfo(cleanedQuery);
                userInputHandler();
                return result;
            } else {
                return 'Please provide a search query.';
            }
        }
    }

    return "I'm not sure how to handle that request.";
}

function userInputHandler(input) {
    return handleUserInput(input)
        .then(result => {
            console.log(result);
            return result;
        })
        .catch(error => {
            console.error('Error handling user input:', error);
            return "Unfortunately an error occurred while handling the input.";
        });
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
    model = createModel(Object.keys(vocab).length);
    await trainModel(model, data, labels);
}

initializeModel();

// API endpoint for chatting
expressApp.post('/chat', async (req, res) => {
    if (!chatEnabled) {
        res.json({ response: "Chat is currently disabled while performing a search. Please try again in a moment." });
        return;
    }
    
    const { message } = req.body;
    const normalizedMessage = normalizeText(message.replace(/[,']/g, "").toLowerCase());

    // Check for Bing search keywords
    if (['search bing', 'bing', 'search bing for'].some(keyword => normalizedMessage.startsWith(keyword))) {
        const bingResponse = await handleUserInput(message);
        res.json({ response: bingResponse });
        return;
    }

    // Prioritize Wikipedia query for specific patterns
    if (isWikipediaQuery(message)) {
        const summarize = /summarize|summary|bullet points/i.test(message);
        let response = await getWikipediaInfo(message, summarize);
        if (response.includes("Sorry, I couldn't find any relevant information on Wikipedia")) {
            response = await getBingSearchInfo(message);
        }
        res.json({ response });
    } else {
        let response = findBestMatch(normalizedMessage, knowledge);
        
        if (response === "Sorry, I didn't understand that.") {
            let wikipediaInfo = await getWikipediaInfo(message);
            if (wikipediaInfo.includes("Sorry, I couldn't find any relevant information on Wikipedia")) {
                wikipediaInfo = await getBingSearchInfo(message);
            }
            res.json({ response: wikipediaInfo });
        } else {
            res.json({ response });
        }
    }

    // Add the interaction to training data
    if (response && !message.startsWith('/')) {
        addTrainingData(normalizedMessage, response);
    }

    res.json({ response });
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
    console.log(`Server startup was successful.`);
});

// Electron App Initialization
let win;

function createWindow() {
    win = new BrowserWindow({
        width: 1250,
        height: 1150,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
            sandbox: true,
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
