const { app, BrowserWindow, ipcMain } = require('electron');
const portfinder = require('portfinder'); // New package to find available ports
const fs = require('fs');
const path = require('path');
const express = require('express');
const bodyParser = require('body-parser');
const https = require('https');
const wiki = require('wikijs').default;

const DEFAULT_PORT = 3000;
let expressApp;
let server;
let win;

// Electron Window
function createWindow() {
    win = new BrowserWindow({
        width: 1250,
        height: 1150,
        webPreferences: {
            nodeIntegration: false, // Set to false for security in production
            contextIsolation: true, // Set to true for security
        },
    });

    win.loadFile(path.join(__dirname, 'public', 'AI_HtWebz_Assistant_Version 0.4.html'));
    win.on('closed', () => {
        win = null;
    });
}

// Function to get the public IP address
function getPublicIP() {
    return new Promise((resolve, reject) => {
        https.get('https://ifconfig.me', (resp) => {
            let data = '';
            resp.on('data', (chunk) => {
                data += chunk;
            });
            resp.on('end', () => {
                resolve(data);
            });
        }).on('error', (err) => {
            reject(err);
        });
    });
}

// Function to start the Express server
async function startServer(port = DEFAULT_PORT) {
    return new Promise(async (resolve, reject) => {
        if (server) {
            console.log('Server is already running.');
            resolve();
            return;
        }

        portfinder.getPort({ port: port, stopPort: port + 100 }, async (err, availablePort) => {
            if (err) {
                reject(err);
                return;
            }

            expressApp = express();
            expressApp.use(bodyParser.json());
            expressApp.use(express.static(path.join(__dirname, 'public')));

            // Initialize knowledge and transitions
            const knowledgeFile = 'knowledge.json';
            const transitionsFile = 'transitions.json';
            let knowledge = {};
            let transitions = {};

            if (fs.existsSync(knowledgeFile)) {
                knowledge = JSON.parse(fs.readFileSync(knowledgeFile, 'utf8'));
            } else {
                fs.writeFileSync(knowledgeFile, JSON.stringify(knowledge, null, 2));
            }

            if (fs.existsSync(transitionsFile)) {
                transitions = JSON.parse(fs.readFileSync(transitionsFile, 'utf8'));
            } else {
                transitions = { default: "Additionally, ", pause: "Meanwhile, " };
                fs.writeFileSync(transitionsFile, JSON.stringify(transitions, null, 2));
            }

            const slangMapping = {
                "idk": "I don't know",
                "brb": "be right back",
                "omg": "Oh my god",
                "nvm": "Never mind",
            };

            function replaceSlang(input) {
                let updatedInput = input.toLowerCase();
                for (const [slang, fullForm] of Object.entries(slangMapping)) {
                    updatedInput = updatedInput.replace(new RegExp(`\\b${slang}\\b`, 'g'), fullForm);
                }
                return updatedInput;
            }

            const categories = {
                greetings: ["hi", "hello", "hey", "morning", "evening"],
                weather: ["weather", "rain", "sunny", "cloudy", "forecast"],
            };

            function matchKeywords(input) {
                const normalizedInput = input.toLowerCase();
                for (const category in categories) {
                    for (const keyword of categories[category]) {
                        if (normalizedInput.includes(keyword)) {
                            return category;
                        }
                    }
                }
                return null;
            }

            function parseSentences(input) {
                return input
                    .split(/(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+/)
                    .map(sentence => sentence.trim())
                    .filter(sentence => sentence.length > 0);
            }

            function processSentence(sentence) {
                const sentenceWithoutSlang = replaceSlang(sentence);
                const normalizedMessage = sentenceWithoutSlang.toLowerCase().replace(/[,']/g, "");
                const category = matchKeywords(normalizedMessage);

                if (category) {
                    switch (category) {
                        case 'greetings':
                            return "Hello! How can I assist you today?";
                        case 'weather':
                            return "I can help with the weather. Where are you located?";
                        default:
                            return "I'm not sure about that. Can you clarify?";
                    }
                }

                return "I don't know how to respond to that.";
            }

            expressApp.post('/chat', (req, res) => {
                const { message } = req.body;
                const sentences = parseSentences(message);
                const responses = sentences.map(sentence => processSentence(sentence));

                const finalResponse = responses.reduce((acc, response, index) => {
                    if (index > 0 && transitions.default) {
                        return `${acc} ${transitions.default} ${response}`;
                    }
                    return `${acc} ${response}`;
                }, "").trim();

                res.json({ response: finalResponse });
            });

            expressApp.post('/feedback', (req, res) => {
                const { message, correctResponse } = req.body;
                const sentences = parseSentences(message);
                sentences.forEach(sentence => {
                    const correctedSentence = replaceSlang(sentence.toLowerCase());
                    const normalizedInput = correctedSentence.replace(/[,']/g, "");

                    knowledge[normalizedInput] = correctResponse;
                    knowledge[sentence.toLowerCase()] = correctResponse;
                });

                fs.writeFileSync(knowledgeFile, JSON.stringify(knowledge, null, 2));
                res.json({ response: "Thank you for your feedback!" });
            });

            expressApp.get('/', (req, res) => {
                res.sendFile(path.join(__dirname, 'public', 'AI_HtWebz_Assistant_Version 0.4.html'));
            });

            server = expressApp.listen(availablePort, '0.0.0.0', async () => {
                const publicIP = await getPublicIP();
                console.log(`Server running and accessible at http://${publicIP}:${availablePort}`);
                resolve();
            });

            server.on('error', (error) => {
                console.error(`Server error: ${error.message}`);
                reject(error);
            });
        });
    });
}

// Electron app startup
//app.whenReady().then(async () => {
//    try {
//        await startServer(); // Ensure the server starts before creating the window
//        createWindow(); // Start the Electron window after server is ready
//    } catch (error) {
//        console.error('Error starting the server:', error);
//    }
//});

// Handling app behavior for macOS
//app.on('activate', () => {
//    if (BrowserWindow.getAllWindows().length === 0) {
//        createWindow();
//    }
//});

// Quit the app when all windows are closed (except macOS)
//app.on('window-all-closed', () => {
//    if (process.platform !== 'darwin') {
//        app.quit();
//    }
//});
