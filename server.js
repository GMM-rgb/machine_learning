const express = require('express');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');
const { app, BrowserWindow } = require('electron');
const wiki = require('wikijs').default;
const math = require('mathjs');

const expressApp = express();
const PORT = 3000;

// Middleware to parse JSON and serve static files
expressApp.use(bodyParser.json());
expressApp.use(express.static(path.join(__dirname, 'public')));

// Normalize input by replacing contractions and slang
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
9
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

// Wikipedia info fetching with input sanitization (e.g., "What is ${normalizedInput}?") 
async function getWikipediaInfo(query) {
    // Clean up punctuation and prepare query
    const sanitizedQuery = query.replace(/[^\w\s]/g, '').trim();

    try {
        const page = await wiki().page(sanitizedQuery);
        const summary = await page.summary();
        return `Here's what I found on Wikipedia: ${summary}`;
    } catch (error) {
        return "Sorry, I couldn't find any relevant information on Wikipedia.";
    }
}

// API endpoint for chatting
expressApp.post('/chat', async (req, res) => {
    const { message } = req.body;

    const normalizedMessage = normalizeText(message.replace(/[,']/g, "").toLowerCase());
    let response = findBestMatch(normalizedMessage, knowledge);
    
    // If no match in knowledge and it's a question like "What is X?" or "Tell me about Y?"
    if (response === "Sorry, I didn't understand that." && isWikipediaQuery(message)) {
        const wikipediaInfo = await getWikipediaInfo(message);
        res.json({ response: wikipediaInfo });
    } else {
        res.json({ response });
    }
});

const ipconfig = '169.199.141.197';

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

// Serve the HTML file
expressApp.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'AI_HtWebz_Assistant_Version 0.4.html'));
});

// Start server
expressApp.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://${ipconfig}:${PORT}`);
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
        },
    });

    win.loadURL(`http://${ipconfig}:${PORT}`);
    win.on('closed', () => {
        win = null;
    });
}

app.whenReady().then(() => {
    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});
