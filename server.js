const express = require("express");
const bodyParser = require("body-parser");
const fs = require("fs");
const path = require("path");
const axios = require("axios");
const wiki = require("wikijs").default;
const math = require("mathjs");
const tf = require("@tensorflow/tfjs-node");
const ResponseGenerator = require('./response_generator');

const expressApp = express();
const PORT = 3002;

let chatEnabled = true;
let model;

const data = [];
const labels = [];
const vocab = {};

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

function saveUserData() {
  fs.writeFileSync(usersFile, JSON.stringify(users, null, 2));
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

if (fs.existsSync(trainingDataFile)) {
  trainingData = JSON.parse(fs.readFileSync(trainingDataFile, "utf8"));
} else {
  fs.writeFileSync(trainingDataFile, JSON.stringify(trainingData, null, 2));
}

function saveTrainingData() {
  trainingData.lastTrainingDate = new Date().toISOString();
  fs.writeFileSync(trainingDataFile, JSON.stringify(trainingData, null, 2));
}

// Initialize ResponseGenerator
const responseGenerator = new ResponseGenerator(
  knowledgeFile,
  trainingDataFile,
  "model/"
);

responseGenerator.currentDateTime = new Date().toISOString();
responseGenerator.currentUser = 'GMM-rgb';

// Conversation data per chat
const conversationData = new Map();

// User registration
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

// User login
expressApp.post("/login", (req, res) => {
  const { username, password } = req.body;

  if (!users[username] || users[username].password !== password) {
    res.json({ success: false, message: "Invalid username or password." });
    return;
  }

  res.json({ success: true, accountId: users[username].accountId });
});

// Normalize input
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

// Wikipedia info fetching
async function getWikipediaInfo(query, previousContext = null) {
  const sanitizedQuery = query
    .toLowerCase()
    .replace(/^(what is|what are|who is|describe|explain|when did|where is|how did)\s+/i, '')
    .replace(/[?.,!]/g, '')
    .trim();

  try {
    let searchQuery = sanitizedQuery;
    if (previousContext) {
      const contextWords = previousContext.split(' ')
        .filter(word => word.length > 3)
        .slice(-3)
        .join(' ');
      searchQuery = `${contextWords} ${sanitizedQuery}`;
    }

    const searchResults = await wiki().search(searchQuery);
    if (!searchResults.results || !searchResults.results.length) {
      return `Sorry, I couldn't find any relevant information about ${query}.`;
    }

    const page = await wiki().page(searchResults.results[0]);
    const [summary, references] = await Promise.all([
      page.summary(),
      page.references().catch(() => [])
    ]);

    let response = summary;

    if (references && references.length > 0) {
      response += `\n\nSource: ${references[0]}`;
    }

    return response;
  } catch (error) {
    console.error(`Error fetching Wikipedia data for "${sanitizedQuery}":`, error);
    return `Sorry, I couldn't find any relevant information about ${query}.`;
  }
}

// Bing search
async function getBingSearchInfo(query) {
  const subscriptionKey = "1feda3372abf425494ce986ad9024238";
  const endpoint = "https://api.bing.microsoft.com/v7.0/search";
  const topCount = process.env.BING_TOP_COUNT || 3;

  try {
    chatEnabled = false;
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
        count: topCount,
        responseFilter: "Webpages",
        mkt: "en-US",
      },
    });

    if (response.data?.webPages?.value && response.data.webPages.value.length > 0) {
      const results = response.data.webPages.value;
      console.log("Bing search results:", results.map(r => r.name));
      let resultText = "Bing Top Results:\n";
      results.forEach((result, idx) => {
        resultText += `${idx + 1}. ${result.name} - ${result.url}\n`;
      });
      return resultText;
    } else {
      console.warn("No Bing search results found for query:", query);
      return "No results found on Bing.";
    }
  } catch (error) {
    console.error("Bing search error:", error.response ? error.response.data : error.message);
    return "Sorry, I couldn't complete the Bing search at this time.";
  } finally {
    chatEnabled = true;
  }
}

// DuckDuckGo results
async function getDuckDuckGoResults(query) {
  try {
    const response = await axios.get('https://api.duckduckgo.com/', {
      params: {
        q: query,
        format: 'json',
        t: 'AIAssistant'
      }
    });

    const results = response.data.RelatedTopics
      .filter(topic => topic.FirstURL && topic.Text)
      .map(topic => ({
        url: topic.FirstURL,
        title: topic.Text.split(' - ')[0],
        snippet: topic.Text.split(' - ').slice(1).join(' - ') || topic.Text,
        source: new URL(topic.FirstURL).hostname.replace(/^www\./, '')
      }))
      .slice(0, 3);

    return results;
  } catch (error) {
    console.error('Error fetching DuckDuckGo results:', error);
    return [];
  }
}

// Related wiki articles
async function findRelatedWikiArticles(topic) {
  try {
    const searchResults = await wiki().search(topic, 5);
    return searchResults.results.map(result => result.title);
  } catch (error) {
    console.error("Error finding related articles:", error);
    return [];
  }
}

// Math problem solving
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

    if (cleanedExpression.includes("!")) {
      const num = parseInt(cleanedExpression.replace("!", ""));
      return `The factorial of ${num} is ${math.factorial(num)}`;
    }

    const result = math.evaluate(cleanedExpression);

    if (math.typeOf(result) === "Matrix") {
      return `Result:\n${result.toString()}`;
    } else if (typeof result === "number") {
      return `The answer is: ${Number.isInteger(result) ? result : result.toFixed(4)}`;
    } else {
      return `Result: ${result.toString()}`;
    }
  } catch (error) {
    console.error("Math evaluation error:", error);
    return "Sorry, I couldn't solve that math problem. Please check the expression and try again.";
  }
}

// UPDATED: Main chat endpoint with improved response handling
expressApp.post("/chat", async (req, res) => {
  if (!chatEnabled) {
    return res.json({
      response: "Chat is currently disabled while performing a search. Please try again in a moment.",
      html: "<div class='system-message'>Chat is currently disabled while performing a search. Please try again in a moment.</div>"
    });
  }

  const { message, accountId = "default", chatId } = req.body;

  if (!message) {
    return res.status(400).json({
      response: "Please provide a message.",
      html: "<div class='error-message'>Please provide a message.</div>"
    });
  }

  try {
    // Initialize conversation data for this chat if it doesn't exist
    if (!conversationData.has(chatId)) {
      conversationData.set(chatId, []);
    }

    // Get chat history
    const chatHistory = conversationData.get(chatId) || [];

    let response = "";
    let cleanedMessage = "";
    let htmlResponse = "";
    const messageForChecks = message.trim().toLowerCase();

    // 1. Math problems
    if (messageForChecks.match(/[\d+\-*/()^√π]|math|calculate|solve|algebra/i)) {
      cleanedMessage = message.replace(/(math|calculate|solve|algebra)/gi, '').trim();
      response = await solveMathProblem(cleanedMessage);
      htmlResponse = `<div class='math-response'>${response}</div>`;

    // 2. Bing search command
    } else if (messageForChecks.startsWith("search bing") || messageForChecks.startsWith("bing")) {
      cleanedMessage = message.replace(/^(search\s+bing|bing)\s*/i, '').trim();
      response = await getBingSearchInfo(cleanedMessage);
      htmlResponse = `<div class='search-response'>
                    <div class='search-title'>Search Results:</div>
                    <div class='search-content'>${response}</div>
                  </div>`;

    // 3. Wiki/Info questions
    } else if (
      messageForChecks.includes("wiki") ||
      messageForChecks.includes("what is") ||
      messageForChecks.includes("who is") ||
      messageForChecks.includes("explain to me") ||
      messageForChecks.includes("explain") ||
      messageForChecks.includes("what are") ||
      messageForChecks.includes("when did") ||
      messageForChecks.includes("where is") ||
      messageForChecks.includes("how did") ||
      messageForChecks.includes("describe")
    ) {
      cleanedMessage = message
        .replace(/^(wiki|what is|who is|what are|describe|explain|explain to me|when did|where is|how did)\s*/i, '')
        .replace(/\?+$/, '')
        .trim();

      let wikiInfo = "";
      try {
        const previousContext = chatHistory.length > 0 ? chatHistory[chatHistory.length - 1].text : null;
        wikiInfo = await getWikipediaInfo(cleanedMessage, previousContext);
      } catch (wikiLookupError) {
        console.error("Wikipedia lookup error:", wikiLookupError);
      }

      // Use enhanced response generator
      const possibilities = await responseGenerator.generateEnhancedResponse(message, chatHistory);
      response = possibilities && possibilities.length > 0 ? possibilities[0].response : wikiInfo;

      let relatedArticlesHtml = "";
      try {
        const relatedArticles = await findRelatedWikiArticles(cleanedMessage);
        if (relatedArticles && relatedArticles.length > 0) {
          relatedArticlesHtml = relatedArticles
            .slice(0, 3)
            .map(article => `• ${article}`)
            .join("\n");
        }
      } catch (err) {
        console.error("Error fetching related wiki articles:", err);
      }

      let webArticles = [];
      try {
        webArticles = await getDuckDuckGoResults(cleanedMessage);
      } catch (err2) {
        console.error("DuckDuckGo search error:", err2);
      }

      htmlResponse = `
        <div class='ai-response'>
          <div class='response-main'>${response}</div>
          
          ${wikiInfo && wikiInfo !== response ? `
            <div class='wiki-section'>
              <h4>Wikipedia Says:</h4>
              <div class='wiki-content'>${wikiInfo}</div>
              ${relatedArticlesHtml ? `
                <div class='related-topics'>
                  <h5>Related Topics:</h5>
                  <pre>${relatedArticlesHtml}</pre>
                </div>
              ` : ""}
            </div>
          ` : ""}
          
          ${webArticles.length > 0 ? `
            <div class='web-references'>
              <h4>Related Articles:</h4>
              <div class='references-grid'>
                ${webArticles.map(article => `
                  <div class='article-card'>
                    <h5>${article.title}</h5>
                    <p class='snippet'>${article.snippet}</p>
                    <div class='article-footer'>
                      <span class='source'>${article.source}</span>
                      <a href="${article.url}" target="_blank" rel="noopener">Read More →</a>
                    </div>
                  </div>
                `).join('')}
              </div>
            </div>
          ` : ""}
        </div>
      `;

    // 4. Normal chat handling with enhanced response generator
    } else {
      const possibilities = await responseGenerator.generateEnhancedResponse(message, chatHistory);
      
      if (possibilities && possibilities.length > 0) {
        response = possibilities[0].response;
        const confidence = (possibilities[0].confidence * 100).toFixed(1);

        let html = `<div class='ai-response'>
          <div class='response-main'>${response}</div>
          <div class='confidence-indicator' style='opacity: 0.6; font-size: 0.85em; margin-top: 8px;'>
            Confidence: ${confidence}% | Source: ${possibilities[0].source}
          </div>
        `;

        // Show alternative responses if available
        if (possibilities.length > 1 && possibilities[1].confidence > 0.5) {
          html += "<div class='alternative-responses' style='margin-top: 12px; padding: 8px; background: #f5f5f5; border-radius: 4px;'>";
          html += "<div style='font-weight: bold; margin-bottom: 6px;'>Alternative perspectives:</div>";
          possibilities.slice(1, 3).forEach((p, index) => {
            html += `<div class='alt-response' style='margin: 4px 0; padding-left: 8px; border-left: 2px solid #ccc;'>
              ${p.response} <span style='opacity: 0.6; font-size: 0.85em;'>(${(p.confidence * 100).toFixed(1)}%)</span>
            </div>`;
          });
          html += "</div>";
        }

        html += "</div>";
        htmlResponse = html;
      } else {
        response = "I'm not quite sure how to respond to that. Could you rephrase or provide more context?";
        htmlResponse = `<div class='ai-response'>${response}</div>`;
      }
    }

    // Fallback for question-like inputs without wiki handling
    if (messageForChecks.match(/^(what|how|why|explain|who|when|where)/i) && !htmlResponse.includes("wiki-section")) {
      try {
        const wikiInfoFallback = await getWikipediaInfo(message);
        const webArticlesFallback = await getDuckDuckGoResults(message);
        
        if (wikiInfoFallback || webArticlesFallback.length > 0) {
          htmlResponse = `
            <div class='ai-response'>
              <div class='response-main'>${response}</div>
              
              ${wikiInfoFallback && wikiInfoFallback !== response ? `
                <div class='wiki-section'>
                  <h4>Additional Information:</h4>
                  <div class='wiki-content'>${wikiInfoFallback}</div>
                </div>
              ` : ""}
              
              ${webArticlesFallback.length > 0 ? `
                <div class='web-references'>
                  <h4>Related Articles:</h4>
                  <div class='references-grid'>
                    ${webArticlesFallback.map(article => `
                      <div class='article-card'>
                        <h5>${article.title}</h5>
                        <p class='snippet'>${article.snippet}</p>
                        <div class='article-footer'>
                          <span class='source'>${article.source}</span>
                          <a href="${article.url}" target="_blank" rel="noopener">Read More →</a>
                        </div>
                      </div>
                    `).join('')}
                  </div>
                </div>
              ` : ""}
            </div>
          `;
        }
      } catch (fallbackError) {
        console.error("Fallback wiki/web search error:", fallbackError);
      }
    }

    // Save conversation history
    chatHistory.push({ sender: 'User', text: message });
    if (response) {
      chatHistory.push({ sender: 'AI', text: response });
    }
    conversationData.set(chatId, chatHistory);

    res.json({ response, html: htmlResponse });
  } catch (error) {
    console.error("Chat error:", error);
    res.status(500).json({ 
      response: "Sorry, I encountered an error.", 
      html: "<div class='error-message'>Sorry, I encountered an error.</div>" 
    });
  }
});

// Feedback endpoint
expressApp.post("/feedback", (req, res) => {
  const { message, correctResponse } = req.body;
  
  if (message.toLowerCase().startsWith('correction:')) {
    const normalizedInput = normalizeText(
      message.replace(/^correction:\s*/i, "").toLowerCase()
    );
    knowledge[normalizedInput] = correctResponse;
    knowledge[message.toLowerCase()] = correctResponse;

    try {
      fs.writeFileSync(knowledgeFile, JSON.stringify(knowledge, null, 2));
      console.log("Knowledge data saved!");
    } catch (err) {
      console.error("Error saving knowledge:", err);
    }
  }
  res.json({ response: "Thank you for your feedback!" });
});

// Get user conversations
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

// Save user conversations
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

// Delete user conversation
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

// Get local IP address
function getLocalIpAddress() {
  const { networkInterfaces } = require("os");
  const nets = networkInterfaces();
  const results = [];
  const targetIP = "192.168.0.62";

  for (const name of Object.keys(nets)) {
    for (const net of nets[name]) {
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
  }
  results.sort((a, b) => b.isMain - a.isMain);
  return results;
}

// Serve main page
expressApp.get("/", (req, res) => {
  res.sendFile(
    path.join(__dirname, "public", "AI_HtWebz_Assistant_Version 0.4.html")
  );
});

// Styles route
expressApp.get("/styles.css", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "styles.css"));
});

// Static files
expressApp.use("/", express.static(path.join(__dirname, "public")));

// Server startup
expressApp.listen(PORT, "0.0.0.0", () => {
  const localIps = getLocalIpAddress();
  console.log("\n=== Server Network Information ===");
  console.log(`Local Access: http://localhost:${PORT}`);
  console.log(`\nNetwork Access URLs:`);

  if (localIps.length > 0) {
    localIps.forEach(({ name, address, isMain }) => {
      if (isMain) {
        console.log(`\n→ Main URL (Your IP): http://${address}:${PORT}`);
        console.log(`  Use this URL to access from other devices on your network`);
      } else {
        console.log(`\nAlternative URL: http://${address}:${PORT}`);
      }
    });
  } else {
    console.log("No network interfaces found");
  }

  console.log("\nServer startup & setup was successful.");
});

console.log("Server.js loaded successfully, and has been initialized.");
