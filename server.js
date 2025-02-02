const express = require("express");
const bodyParser = require("body-parser");
const fs = require("fs");
const path = require("path");
const axios = require("axios");
const { app, BrowserWindow } = require("electron");
const wiki = require("wikijs").default;
const math = require("mathjs");
const { ifError } = require("assert");
const tf = require("@tensorflow/tfjs-node-gpu");
const https = require("https");
const ProgressBar = require("progress");
const cors = require("cors");
const os = require("os");

const expressApp = express();
const PORT = 3000;

let chatEnabled = true; // Flag to control chat availability

// Middleware to parse JSON and serve static files
expressApp.use(bodyParser.json());
expressApp.use(express.static(path.join(__dirname, "public")));

// Load or initialize user data
const usersFile = path.join(__dirname, "users.json");
let users = {};

if (fs.existsSync(usersFile)) {
  users = JSON.parse(fs.readFileSync(usersFile, "utf8"));
} else {
  fs.writeFileSync(usersFile, JSON.stringify(users, null, 2));
}

// Function to save user data
function saveUserData() {
  fs.writeFileSync(usersFile, JSON.stringify(users, null, 2));
}

// Endpoint for user registration
expressApp.post("/signup", (req, res) => {
  const { username, password } = req.body;

  if (users[username]) {
    res.json({ success: false, message: "Username already exists." });
    return;
  }

  users[username] = { password, accountId: `account_${Date.now()}` };
  saveUserData();

  res.json({ success: true, accountId: users[username].accountId });
});

// Endpoint for user login
expressApp.post("/login", (req, res) => {
  const { username, password } = req.body;

  if (!users[username] || users[username].password !== password) {
    res.json({ success: false, message: "Invalid username or password." });
    return;
  }

  res.json({ success: true, accountId: users[username].accountId });
});

// Normalize input by replacing contractions and short term meaning
function normalizeText(input) {
  const maps = {
    contractions: {
      its: "it's",
      im: "i'm",
      youre: "you're",
      theyre: "they're",
      were: "we're",
      hes: "he's",
      shes: "she's",
      thats: "that is",
      cant: "cannot",
      dont: "do not",
      doesnt: "does not",
      wont: "will not",
      isnt: "is not",
      arent: "are not",
      werent: "were not",
      hasnt: "has not",
      havent: "have not",
      didnt: "did not",
      wouldnt: "would not",
      couldnt: "could not",
      shouldnt: "should not",
      mightnt: "might not",
    },
    slang: {
      idk: "I don't know",
      idr: "I don't remember",
      omg: "Oh my God",
      btw: "By the way",
      lol: "Laugh out loud",
      brb: "Be right back",
      gtg: "Got to go",
      ttyl: "Talk to you later",
      fyi: "For your information",
      smh: "Shaking my head",
      lmao: "Laughing my ass off",
      bff: "Best friends forever",
      tbh: "To be honest",
      yolo: "You only live once",
      nvm: "Never mind",
      ty: "Thank you",
      yw: "Your welcome",
    },
  };

  const words = input.split(/\s+/);
  return words
    .map(
      (word) =>
        maps.contractions[word.toLowerCase()] ||
        maps.slang[word.toLowerCase()] ||
        word
    )
    .join(" ");
}

// Load or initialize knowledge
const knowledgeFile = path.join(__dirname, "knowledge.json");
let knowledge = {};

if (fs.existsSync(knowledgeFile)) {
  knowledge = JSON.parse(fs.readFileSync(knowledgeFile, "utf8"));
} else {
  fs.writeFileSync(knowledgeFile, JSON.stringify(knowledge, null, 2));
}

// Initialize training data storage
const trainingDataFile = path.join(__dirname, "training_data.json");
let trainingData = {
  conversations: [],
  vocabulary: {},
  lastTrainingDate: null,
};

// Load or create training data file
if (fs.existsSync(trainingDataFile)) {
  trainingData = JSON.parse(fs.readFileSync(trainingDataFile, "utf8"));
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
    timestamp: new Date().toISOString(),
  });

  // Update vocabulary
  const words = preprocessText(userMessage + " " + aiResponse);
  words.forEach((word) => {
    if (!trainingData.vocabulary[word]) {
      trainingData.vocabulary[word] =
        Object.keys(trainingData.vocabulary).length + 1;
    }
  });

  saveTrainingData();

  // Retrain model if we have enough new data (every 5 conversations)
  if (trainingData.conversations.length % 5 === 0) {
    retrainModel();
  }
}

// Function to read training data in a special encoding
function readTrainingData() {
  if (fs.existsSync(trainingDataFile)) {
    const rawData = fs.readFileSync(trainingDataFile, "utf8");
    const parsedData = JSON.parse(rawData);

    // Process the training data in a special encoding
    parsedData.conversations.forEach((conversation) => {
      const inputWords = preprocessText(conversation.input);
      const outputWords = preprocessText(conversation.output);

      // Create training pairs
      inputWords.forEach((word, index) => {
        if (!vocab[word]) {
          vocab[word] = Object.keys(vocab).length + 1;
        }
        if (index < inputWords.length - 1) {
          data.push(inputWords.slice(0, index + 1).map((w) => vocab[w]));
          labels.push([vocab[inputWords[index + 1]]]);
        }
      });

      // Also learn from AI responses
      outputWords.forEach((word, index) => {
        if (!vocab[word]) {
          vocab[word] = Object.keys(vocab).length + 1;
        }
        if (index < outputWords.length - 1) {
          data.push(outputWords.slice(0, index + 1).map((w) => vocab[w]));
          labels.push([vocab[outputWords[index + 1]]]);
        }
      });
    });

    trainingData = parsedData;
  }
}

// Initialize vocabulary and model
const vocab = {};
let model;
const data = [];
const labels = [];

// Train the model with existing knowledge
async function initializeModel() {
  readTrainingData();

  // Add knowledge base data
  for (const key in knowledge) {
    const words = preprocessText(key);
    words.forEach((word, index) => {
      if (!vocab[word]) {
        vocab[word] = Object.keys(vocab).length + 1;
      }
      if (index < words.length - 1) {
        data.push(words.slice(0, index + 1).map((w) => vocab[w]));
        labels.push([vocab[words[index + 1]]]);
      }
    });
  }

  model = createTransformerModel(Object.keys(vocab).length);
  await trainTransformerModel(model, data, labels);
}

// Function to retrain the model with accumulated data
async function retrainModel() {
  const data = [];
  const labels = [];

  // Process all conversations for training
  trainingData.conversations.forEach((conversation) => {
    const inputWords = preprocessText(conversation.input);
    const outputWords = preprocessText(conversation.output);

    // Create training pairs
    inputWords.forEach((word, index) => {
      if (!vocab[word]) {
        vocab[word] = Object.keys(vocab).length + 1;
      }
      if (index < inputWords.length - 1) {
        data.push(inputWords.slice(0, index + 1).map((w) => vocab[w]));
        labels.push([vocab[inputWords[index + 1]]]);
      }
    });

    // Also learn from AI responses
    outputWords.forEach((word, index) => {
      if (!vocab[word]) {
        vocab[word] = Object.keys(vocab).length + 1;
      }
      if (index < outputWords.length - 1) {
        data.push(outputWords.slice(0, index + 1).map((w) => vocab[w]));
        labels.push([vocab[outputWords[index + 1]]]);
      }
    });
  });

  // Retrain the model
  model = createModel(Object.keys(vocab).length);
  await trainModel(model, data, labels);
  console.log("Model retrained with accumulated data");
}

function getLevenshteinDistance(a, b) {
  const tmp = Array(b.length + 1)
    .fill(null)
    .map(() => Array(a.length + 1).fill(0));

  for (let i = 0; i <= b.length; i++) {
    tmp[i][0] = i;
  }
  for (let j = 0; j <= a.length; j++) {
    tmp[0][j] = j;
  }

  for (let i = 1; i <= b.length; i++) {
    for (let j = 1; j <= a.length; j++) {
      tmp[i][j] =
        b[i - 1] === a[j - 1]
          ? tmp[i - 1][j - 1]
          : Math.min(tmp[i - 1][j - 1] + 1, tmp[i][j - 1] + 1, tmp[i - 1][j] + 1);
    }
  }
  tmp[b.length][a.length]; // Final distance
  console.log("Levenshtein Distance:", tmp[b.length][a.length]);
  if (a.length > 1000 || b.length > 1000) {
    console.warn("Strings are too long, skipping computation.");
    return -1;
  }
}

// Find the best match in knowledge (with exact and fuzzy match)
function findBestMatch(query, knowledge) {
  let closestMatch = null;
  let minDistance = Infinity;

  // Check for exact matches first
  for (const key in (knowledge)) {
    const normalizedKey = key.toLowerCase();
    if (query === normalizedKey) {
      return knowledge[key];
    }
  }

  // Use fuzzy matching based on Levenshtein distance
  for (const key in (knowledge)) {
    const distance = getLevenshteinDistance(query, key);
    if (distance < minDistance) {
      minDistance = distance;
      closestMatch = key;
    }
  }

  return closestMatch
    ? knowledge[closestMatch]
    : "Sorry, I didn't understand that.";
}

// Detect if the user is asking for Wikipedia information
function isWikipediaQuery(input) {
  const wikipediaTriggers = [
    /^what is |^what are |^what was |^tell me about |^who is |^explain |^define |^describe /i,
    /^how does |^how do |^how can |^what does |^where is |^when did /i,
  ];

  const normalizedInput = input.toLowerCase().trim();
  return wikipediaTriggers.some((trigger) => trigger.test(normalizedInput));
}

// Function to summarize text into bullet points
function summarizeText(text) {
  const sentences = text.split(". ");
  const summary = sentences
    .slice(0, 5)
    .map((sentence) => `• ${sentence.trim()}`)
    .join("\n");
  return summary;
}

// Wikipedia info fetching with input sanitization (e.g., "What is X?")
async function getWikipediaInfo(query, summarize = false) {
  // Clean up query
  const sanitizedQuery = query
    .toLowerCase()
    .replace(
      /^(what is|what are|what was|tell me about|who is|explain|define|describe|how does|how do|how can|what does|where is|when did)\s+/i,
      ""
    )
    .replace(/[?.,!]/g, "")
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
      return `Here's a summary of what I found on Wikipedia:\n${summarizeText(
        summary
      )}`;
    }
    return `Here's what I found on Wikipedia: ${summary}`;
  } catch (error) {
    console.error(
      `Error fetching data from Wikipedia for query "${sanitizedQuery}":`,
      error
    );
    return `Sorry, I couldn't find any relevant information on Wikipedia about ${query}.`;
  }
}

// Update Bing search function with better error handling and logging
async function getBingSearchInfo(query) {
  // Need to get a valid API key from Microsoft Azure
  const subscriptionKey = "1feda3372abf425494ce986ad9024238";
  const endpoint = "https://api.bing.microsoft.com/v7.0/search";

  try {
    chatEnabled = false; // Disable chat while searching
    console.log("Initiating Bing search for:", query);

    const response = await axios({
      method: "get",
      url: endpoint,
      headers: {
        "Ocp-Apim-Subscription-Key": subscriptionKey,
        Accept: "application/json",
      },
      params: {
        q: query,
        count: 1,
        responseFilter: "Webpages",
        mkt: "en-US",
      },
    });

    console.log("Bing API response status:", response.status);

    if (
      response.data &&
      response.data.webPages &&
      response.data.webPages.value &&
      response.data.webPages.value.length > 0
    ) {
      const result = response.data.webPages.value[0];
      chatEnabled = true;
      return `Here's what I found on Bing: ${result.name}\n${result.snippet}\nSource: ${result.url}`;
    } else {
      console.log("No results found in Bing response:", response.data);
      return "Sorry, I couldn't find any relevant information on Bing.";
    }
  } catch (error) {
    console.error(
      "Bing search error:",
      error.response ? error.response.data : error.message
    );
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
  const searchKeywords = ["search bing", "bing"];
  const lowerCaseInput = input.toLowerCase().trim();

  // Check if input starts with any search keywords
  for (const keyword of searchKeywords) {
    if (lowerCaseInput.startsWith(keyword)) {
      // Extract the actual search query
      const query = lowerCaseInput.replace(keyword, "").trim();

      if (!query) {
        return "Please provide a search query.";
      }

      console.log("Processing Bing search for:", query);
      return await getBingSearchInfo(query);
    }
  }

  return "I'm not sure how to handle that request.";
}

// Function to preprocess text for TensorFlow (AI) model
function preprocessText(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, "")
    .split(" ");
}

// Function to create a TensorFlow (AI) model
function createModel(vocabSize) {
  const model = tf.sequential();
  model.add(tf.layers.embedding({ inputDim: vocabSize, outputDim: 128 }));
  model.add(tf.layers.lstm({ units: 128, returnSequences: true }));
  model.add(tf.layers.lstm({ units: 128 }));
  model.add(tf.layers.dense({ units: vocabSize, activation: "softmax" }));
  model.compile({ optimizer: "adam", loss: "sparseCategoricalCrossentropy" });
  return model;
}

// Function to train the TensorFlow (AI) model
async function trainModel(model, data, labels, epochs = 10, batchSize = 32) {
// Ensure 'data' is properly formatted
const xs = tf.tensor3d(
  data.map(seq => seq.map(step => [step])), // Reshape to (batch_size, timesteps, features)
  [data.length, data[0].length, 1]
);
  const ys = tf.tensor2d(labels, [labels.length, labels[0].length]); // Ensures labels have the correct shape
  const dataset = tf.data
    .zip({ xs: tf.data.array(xs), ys: tf.data.array(ys) })
    .batch(batchSize);
  await model.fitDataset(dataset, { epochs });
}

// Function to create a Transformer model
function createTransformerModel(vocabSize) {
  const model = tf.sequential();
  model.add(tf.layers.embedding({ inputDim: vocabSize, outputDim: 128 }));
  model.add(tf.layers.dense({ units: 128, activation: "relu" }));
  model.add(tf.layers.dense({ units: vocabSize, activation: "softmax" }));
  model.compile({ optimizer: "adam", loss: "sparseCategoricalCrossentropy" });
  return model;
}

// Function to train the Transformer model
async function trainTransformerModel(model, data, labels, epochs = 10, batchSize = 32) {
  console.log("Validating training data...");

  // Ensure data is correctly formatted as [batch_size, timesteps, features]
  const reshapedData = data.map(seq => seq.map(step => [step])); // Wrap each number in an array

  if (!Array.isArray(reshapedData) || reshapedData.length === 0) {
    throw new Error("Invalid data format: Data must be a non-empty array.");
  }

  if (!Array.isArray(reshapedData[0]) || reshapedData[0].length === 0) {
    throw new Error("Invalid data format: Each data entry must be a non-empty array.");
  }

  if (!Array.isArray(reshapedData[0][0]) || typeof reshapedData[0][0][0] !== "number") {
    throw new Error("Invalid data format: Each inner element must be an array of numbers.");
  }

  console.log(`Data shape: [${reshapedData.length}, ${reshapedData[0].length}, 1]`);
  console.log(`Labels shape: [${labels.length}, ${labels[0].length}]`);

  // Convert data to tensors
  const xs = tf.tensor3d(reshapedData, [reshapedData.length, reshapedData[0].length, 1]);
  const ys = tf.tensor2d(labels, [labels.length, labels[0].length]); // Ensure correct shape

  // Create dataset
  const dataset = tf.data
    .zip({ xs: tf.data.array(xs), ys: tf.data.array(ys) })
    .batch(batchSize);

  console.log("Starting model training...");

  const bar = new ProgressBar("Training [:bar] :percent :etas", {
    total: epochs,
    width: 30,
  });

  await model.fitDataset(dataset, {
    epochs,
    callbacks: {
      onEpochBegin: (epoch) => {
        bar.tick(0, { epoch: epoch + 1 });
      },
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch + 1}: loss=${logs.loss.toFixed(6)}`);
        bar.tick();
      },
    },
  });

  console.log("Model training complete.");
}

// Function to predict the next word using the Transformer model
async function predictNextWordTransformer(model, inputText, vocab) {
  const input = preprocessText(inputText);
  const inputTensor = tf.tensor3d(
    [input.map((word) => [vocab[word] || 0])],
    [1, input.length, 1] // Ensure 3D shape
  );
  const prediction = model.predict(inputTensor);
  const predictedIndex = prediction.argMax(-1).dataSync()[0];
  return Object.keys(vocab).find((key) => vocab[key] === predictedIndex);
}

// Train the model with existing knowledge
async function initializeModel() {
  readTrainingData();

  // Add knowledge base data
  for (const key in knowledge) {
    const words = preprocessText(key);
    words.forEach((word, index) => {
      if (!vocab[word]) {
        vocab[word] = Object.keys(vocab).length + 1;
      }
      if (index < words.length - 1) {
        data.push(words.slice(0, index + 1).map((w) => vocab[w]));
        labels.push([vocab[words[index + 1]]]);
      }
    });
  }

  model = createTransformerModel(Object.keys(vocab).length);
  await trainTransformerModel(model, data, labels);
}

// Function to train the model asynchronously
async function trainModelAsync() {
  console.log("Training model asynchronously...");
  await initializeModel();
  console.log("Model training completed.");

  // Test the prediction function
  const testInput = "hello";
  const predictedWord = await predictNextWordTransformer(
    model,
    testInput,
    vocab
  );
  console.log(`Predicted next word for "${testInput}": ${predictedWord}`);
}

// Main server startup
expressApp.listen(PORT, "0.0.0.0", () => {
  const localIps = getLocalIpAddress();
  console.log("\n=== Server Network Information ===");
  console.log(`Local Access: http://localhost:${PORT}`);
  console.log(`\nNetwork Access URLs:`);

  if (localIps.length > 0) {
    localIps.forEach(({ name, address, isMain }) => {
      if (isMain) {
        console.log(`\n→ Main URL (Your IP): http://${address}:${PORT}`);
        console.log(
          `  Use this URL to access from other devices on your network`
        );
      } else {
        console.log(`\nAlternative URL: http://${address}:${PORT}`);
      }
    });
  } else {
    console.log("No network interfaces found");
  }

  console.log("\nServer startup & setup was successful.");

  // Train the model asynchronously
  trainModelAsync().catch((error) => {
    console.error("Error during model training:", error);
    console.log("Data Type:", typeof data); // Should be 'object'
    console.log("Data Length:", data.length); // Should be greater than 0
    console.log("First element type:", typeof data[0]); // Should be 'object'
    console.log("First element length:", data[0]?.length); // Should be greater than 0
    console.log("First sub-element type:", typeof data[0]?.[0]); // Should be 'object'
    console.log("First sub-sub-element type:", typeof data[0]?.[0]?.[0]); // Should be 'number'
  });
});

// Find similar conversations from training history with fuzzy matching
function findSimilarConversation(input) {
  const conversations = trainingData.conversations;
  let bestMatch = null;
  let bestScore = 0;
  let minDistance = Infinity;

  // Normalize input for comparison
  const normalizedInput = input.toLowerCase().trim();

  for (const conv of conversations) {
    // Try exact word matching first
    const similarity = calculateSimilarity(
      normalizedInput,
      conv.input.toLowerCase()
    );
    if (similarity > bestScore && similarity > 0.6) {
      bestScore = similarity;
      bestMatch = conv;
      continue;
    }

    // If no good word match, try fuzzy matching
    const distance = getLevenshteinDistance(
      normalizedInput,
      conv.input.toLowerCase()
    );
    if (distance < minDistance) {
      minDistance = distance;
      // Only use fuzzy match if it's close enough (adjust threshold as needed)
      if (distance < normalizedInput.length * 0.4) {
        // 40% similarity threshold
        bestMatch = conv;
      }
    }
  }

  return bestMatch;
}

// Enhanced similarity calculation that considers partial matches
function calculateSimilarity(str1, str2) {
  const words1 = str1.split(" ");
  const words2 = str2.split(" ");
  let matches = 0;
  let totalWords = Math.max(words1.length, words2.length);

  // Check each word from the first string
  for (const word1 of words1) {
    // Look for exact or fuzzy matches in the second string
    for (const word2 of words2) {
      if (word1 === word2) {
        matches += 1; // Full match
        break;
      }
      const distance = getLevenshteinDistance(word1, word2);
      if (distance <= Math.min(word1.length, word2.length) * 0.3) {
        // 30% difference tolerance
        matches += 0.8; // Partial match
        break;
      }
    }
  }

  return matches / totalWords;
}

// Get response from training data or knowledge base
function getResponse(message) {
  // First try to find a similar conversation using fuzzy matching
  const similarConversation = findSimilarConversation(message);
  if (similarConversation) {
    console.log("Found similar conversation:", {
      input: similarConversation.input,
      confidence: calculateSimilarity(message, similarConversation.input),
    });
    return similarConversation.output;
  }

  // If no conversation match, try knowledge base
  return findBestMatch(message, knowledge);
}

// Add math detection and solving functions
function isMathQuery(input) {
  const mathTriggers = [
    /^calculate/i,
    /^solve/i,
    /^compute/i,
    /^evaluate/i,
    /=\?$/,
    /\d+[\+\-\*\/\^\(\)]/,
    /[\+\-\*\/\^\(\)]\d+/,
    /\d+\s*[\+\-\*\/\^]\s*\d+/,
  ];

  return mathTriggers.some((trigger) => trigger.test(input));
}

function cleanMathExpression(input) {
  return input
    .toLowerCase()
    .replace(/(calculate|solve|compute|evaluate)/g, "")
    .replace(/[?=]/g, "")
    .replace(/×/g, "*")
    .replace(/÷/g, "/")
    .replace(/\s+/g, "")
    .trim();
}

async function solveMathProblem(input) {
  try {
    const cleanedExpression = cleanMathExpression(input);
    console.log("Solving math expression:", cleanedExpression);

    // Handle special cases
    if (cleanedExpression.includes("!")) {
      const num = parseInt(cleanedExpression.replace("!", ""));
      return `The factorial of ${num} is ${math.factorial(num)}`;
    }

    // Parse and evaluate the expression
    const result = math.evaluate(cleanedExpression);

    // Format the result based on its type
    if (math.typeOf(result) === "Matrix") {
      return `Result:\n${result.toString()}`;
    } else if (typeof result === "number") {
      return `The answer is: ${
        Number.isInteger(result) ? result : result.toFixed(4)
      }`;
    } else {
      return `Result: ${result.toString()}`;
    }
  } catch (error) {
    console.error("Math evaluation error:", error);
    return "Sorry, I couldn't solve that math problem. Please check the expression and try again.";
  }
}

// Add definitions lookup and processing
function findDefinitionInTrainingData(word) {
  if (trainingData.vocabulary.definitions) {
    const definition = trainingData.vocabulary.definitions.find(
      (def) => def.word.toLowerCase() === word.toLowerCase()
    );
    return definition ? definition.definition : null;
  }
  return null;
}

// Enhanced understanding using definitions
function understandInput(input) {
  const words = input.toLowerCase().split(/\s+/);
  const understanding = {
    definitions: [],
    unknownWords: [],
    context: {},
  };

  words.forEach((word) => {
    const definition = findDefinitionInTrainingData(word);
    if (definition) {
      understanding.definitions.push({ word, definition });
    } else {
      understanding.unknownWords.push(word);
    }
  });

  return understanding;
}

// Background learning function
async function learnInBackground(unknownWords) {
  for (const word of unknownWords) {
    try {
      // Try to find information about unknown words
      const info = await getWikipediaInfo(word);
      if (!info.includes("Sorry")) {
        // Add new definition to training data
        if (!trainingData.vocabulary.definitions) {
          trainingData.vocabulary.definitions = [];
        }
        trainingData.vocabulary.definitions.push({
          word: word,
          definition: info.substring(0, info.indexOf(".") + 1),
        });
        saveTrainingData();
        console.log(`Learned new word: ${word}`);
      }
    } catch (error) {
      console.log(`Failed to learn about: ${word}`);
    }
  }
}

// Add conversation history tracking
const conversationHistory = new Map(); // Store conversation history per account

// Function to get varied response based on repetition
function getVariedResponse(message, accountId) {
  if (!conversationHistory.has(accountId)) {
    conversationHistory.set(accountId, []);
  }

  const history = conversationHistory.get(accountId);
  const repeatedCount = history.filter(
    (m) => m.toLowerCase() === message.toLowerCase()
  ).length;

  // Add message to history
  history.push(message);
  // Keep last 10 messages only
  if (history.length > 10) history.shift();

  // If message is repeated, provide variation
  if (repeatedCount > 0) {
    const similarConversation = findSimilarConversation(message);
    if (similarConversation) {
      const variations = [
        `You've already mentioned "${similarConversation.input}". How can I assist further?`,
        `We talked about "${similarConversation.input}" earlier. Anything else on your mind?`,
        `You mentioned "${similarConversation.input}" before. Let's discuss something new.`,
        `I remember you said "${similarConversation.input}". What else would you like to know?`,
      ];
      return variations[repeatedCount % variations.length];
    }
  }

  // If not repeated or no specific variation, return normal response
  return null;
}

// Load or initialize goal data
const goalDataFile = "goal_data.json";
let goalData = {};

if (fs.existsSync(goalDataFile)) {
  goalData = JSON.parse(fs.readFileSync(goalDataFile, "utf8"));
} else {
  fs.writeFileSync(goalDataFile, JSON.stringify(goalData, null, 2));
}

// Function to read goal data
function readGoalData() {
  if (fs.existsSync(goalDataFile)) {
    const rawData = fs.readFileSync(goalDataFile, "utf8");
    goalData = JSON.parse(rawData);
    goalInfo = JSON.parse(rawData);
    console.log("Current Goal:", goalData.goal);
    console.log("Priority:", goalData.priority);
    console.log("Goal retrieval output: ", goalInfo);
  }
}

// Function to get the current goal and priority
function getCurrentGoal() {
  readGoalData();
  return goalData;
}

// Update the chat endpoint to use varied responses and isolate accounts
expressApp.post("/chat", async (req, res) => {
  if (!chatEnabled) {
    res.json({
      response:
        "Chat is currently disabled while performing a search. Please try again in a moment.",
    });
    return;
  }

  const { message, accountId = "default" } = req.body;
  const normalizedMessage = normalizeText(
    message.replace(/[,']/g, "").toLowerCase()
  );
  let response = "";

  try {
    // Check for repeated messages first
    const variedResponse = getVariedResponse(normalizedMessage, accountId);
    if (variedResponse) {
      response = variedResponse;
    } else {
      // Original response logic
      const understanding = understandInput(normalizedMessage);
      console.log("Understanding:", understanding);

      if (understanding.unknownWords.length > 0) {
        learnInBackground(understanding.unknownWords);
      }

      if (isMathQuery(message)) {
        console.log("Math query detected:", message);
        response = await solveMathProblem(message);
      } else if (
        ["search bing", "bing"].some((keyword) =>
          normalizedMessage.startsWith(keyword)
        )
      ) {
        console.log("Bing search requested:", message);
        response = await handleUserInput(message);
      } else if (isWikipediaQuery(message)) {
        console.log("Wikipedia query detected:", message);
        const summarize = /summarize|summary|bullet points/i.test(message);
        response = await getWikipediaInfo(message, summarize);

        if (
          response.includes(
            "Sorry, I couldn't find any relevant information on Wikipedia"
          )
        ) {
          console.log("Wikipedia failed, trying Bing fallback");
          response = await getBingSearchInfo(message);
        }
      } else if (
        ["hi", "hello", "hey", "yo", "sup"].includes(normalizedMessage)
      ) {
        response = "Hello there! How may I help you?";
      } else if (
        normalizedMessage.includes("who is your developer") ||
        normalizedMessage.includes("who created you")
      ) {
        response = "My developer is Maximus Farvour.";
      } else {
        response = getResponse(normalizedMessage);
      }
    }

    // Add the interaction to training data if it's not a command
    if (!message.startsWith("/")) {
      addTrainingData(normalizedMessage, response);
    }

    // Include the current goal and priority in the response
    const currentGoal = getCurrentGoal();
    response += `\n\nCurrent Goal: ${currentGoal.goal}\nPriority: ${currentGoal.priority}`;

    res.json({ response });
  } catch (error) {
    console.error("Error in chat endpoint:", error);
    res
      .status(500)
      .json({ response: "An error occurred while processing your message." });
  }
});

// Feedback API endpoint
expressApp.post("/feedback", (req, res) => {
  const { message, correctResponse } = req.body;

  const normalizedInput = normalizeText(
    message.replace(/[,']/g, "").toLowerCase()
  );
  knowledge[normalizedInput] = correctResponse;
  knowledge[message.toLowerCase()] = correctResponse;

  try {
    fs.writeFileSync(knowledgeFile, JSON.stringify(knowledge, null, 2));
    console.log("Knowledge data saved!");
  } catch (err) {
    console.error("Error saving knowledge:", err);
  }

  res.json({ response: "Thank you for your feedback!" });
});

// Try to start local server for AI when /start is posted from HTML JavaScript
const startManualPORT = 53483;

// Updated the getLocalIpAddress function to highlight your specific IP
function getLocalIpAddress() {
  const { networkInterfaces } = require("os");
  const nets = networkInterfaces();
  const results = [];
  const targetIP = "192.168.0.62";

  for (const name of Object.keys(nets))
    for (const net of nets[name]) {
      // Skip over non-IPv4 and internal (i.e. 127.0.0.1) addresses
      if (net.family === "IPv4" && !net.internal) {
        if (net.address === targetIP) {
          console.log("\n=== Your Main Network Interface ===");
          console.log(`Interface: ${name}`);
          console.log(`IP Address: ${net.address} (This is your machine)`);
          console.log(`Netmask: ${net.netmask}`);
        }
        results.push({
          name: name,
          address: net.address,
          netmask: net.netmask,
          isMain: net.address === targetIP,
        });
      }
    }
  // Sort results to put your IP first
  results.sort((a, b) => b.isMain - a.isMain);
  return results;
}

// Move the route definition before the server startup
expressApp.get("/", (req, res) => {
  res.sendFile(
    path.join(__dirname, "public", "AI_HtWebz_Assistant_Version 0.4.html")
  );
});

// Update the manual server start
expressApp.post("/start", async (req, res) => {
  const localIps = getLocalIpAddress();

  const manualServer = express();
  manualServer.use(express.static(path.join(__dirname, "public")));

  manualServer.get("/", (req, res) => {
    res.sendFile(
      path.join(__dirname, "public", "AI_HtWebz_Assistant_Version 0.4.html")
    );
  });

  manualServer.listen(startManualPORT, "0.0.0.0", () => {
    console.log("\n=== Manual Server Network Information ===");
    console.log(`Local Access: http://localhost:${startManualPORT}`);
    console.log(`\nNetwork Access:`);

    if (localIps.length > 0) {
      localIps.forEach(({ name, address, netmask }) => {
        console.log(`\nInterface: ${name}`);
        console.log(`URL: http://${address}:${startManualPORT}`);
        console.log(`Netmask: ${netmask}`);
      });
    } else {
      console.log("No network interfaces found");
    }
  });

  res.json({ status: "Manual server started" });
});

// Update main server startup with static file serving
expressApp.use("/", express.static(path.join(__dirname, "public")));

// Enable CORS for external access
//expressApp.use(cors());
//expressApp.use(bodyParser.json());
//expressApp.use(express.static(path.join(__dirname, "public")));

// Function to get local IP address
//function getLocalIpAddresses() {
//const interfaces = require("os").networkInterfaces();
//const results = [];

//for (const name of Object.keys(interfaces)) {
//for (const net of interfaces[name]) {
//if (net.family === "IPv4" && !net.internal) {
//results.push({ name, address: net.address });
//}
//}
//}

//return results.length > 0 ? results : [{ name: "localhost", address: "127.0.0.1" }];
//}
//const externalPort = 3001;
// Start server and allow external access
//expressApp.listen(PORT, "0.0.0.0", () => {
//const localIps = getLocalIpAddresses();

//localIps.forEach(({ name, address }) => {
//  console.log(`- Network (${name}): http://${address}:${externalPort}`);
//});

//if (process.env.CODESPACE_NAME) {
//  console.log(`- GitHub Codespaces: https://${process.env.CODESPACE_NAME}-${externalPort}.githubpreview.dev`);
//}
//});

// Electron App Initialization
//let win;

// Update createWindow to use the first valid network interface
//function createWindow() {
//const localIps = getLocalIpAddress();
//win = new BrowserWindow({
//width: 1250,
//height: 1150,
//webPreferences: {
//nodeIntegration: true,
//contextIsolation: false,
//},
//});

//const serverUrl =
//  localIps.length > 0
//   ? `http://${localIps[0].address}:${PORT}`
//   : `http://localhost:${PORT}`;

//win.loadURL(serverUrl).catch((error) => {
//console.error("Failed to load URL:", error);
//});

//win.on("closed", () => {
//  win = null;
//});

//console.log(`\nElectron app loading from: ${serverUrl}`);
//}

//app
//.whenReady()
//.then(() => {
//createWindow();

//app.on("activate", () => {
//if (BrowserWindow.getAllWindows().length === 0) {
//createWindow();
//console.log("BrowserWindow created successfully.");
//}
//});
//})
//.catch((error) => {
//  console.error("Error creating BrowserWindow:", error);
//});

//app.on("window-all-closed", () => {
//if (process.platform !== "darwin") {
//app.quit();
//}
//});

console.log("Server.js loaded successfully, and has been initialized.");

// Verification check to ensure all necessary variables and functions are initialized
if (
  !expressApp ||
  !PORT ||
  !chatEnabled ||
  !usersFile ||
  !users ||
  !knowledgeFile ||
  !knowledge ||
  !trainingDataFile ||
  !trainingData ||
  !vocab ||
  !model ||
  !data ||
  !labels
) {
  console.error(
    "Initialization error: One or more necessary variables or functions are not initialized."
  );
}

// Endpoint to get user conversations
expressApp.post("/getConversations", (req, res) => {
  const { accountId } = req.body;

  const user = Object.values(users).find(
    (user) => user.accountId === accountId
  );
  if (!user) {
    return res.json({ success: false, message: "User not found." });
  }

  res.json({ success: true, conversations: user.conversations || {} });
});

// Endpoint to save user conversations
expressApp.post("/saveConversation", (req, res) => {
  const { accountId, chatId, conversation } = req.body;

  const user = Object.values(users).find(
    (user) => user.accountId === accountId
  );
  if (!user) {
    return res.json({ success: false, message: "User not found." });
  }

  if (!user.conversations) {
    user.conversations = {};
  }

  user.conversations[chatId] = conversation;
  saveUserData();
  res.json({ success: true });
});

// Endpoint to delete user conversation
expressApp.post("/deleteConversation", (req, res) => {
  const { accountId, chatId } = req.body;

  const user = Object.values(users).find(
    (user) => user.accountId === accountId
  );
  if (!user) {
    return res.json({ success: false, message: "User not found." });
  }

  if (user.conversations && user.conversations[chatId]) {
    delete user.conversations[chatId];
    saveUserData();
    return res.json({ success: true });
  }

  res.json({ success: false, message: "Conversation not found." });
});
