// server.js
const express = require("express");
const bodyParser = require("body-parser");
const fs = require("fs");
const path = require("path");
const axios = require("axios");
const wiki = require("wikijs").default;
const math = require("mathjs");
const tf = require("@tensorflow/tfjs-node-gpu");
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
  // Respect environment flag to prevent uncontrolled overwrites.
  // To enable automatic saving, set ALLOW_AUTOSAVE=true in the environment.
  if (process.env.ALLOW_AUTOSAVE !== "true") {
    console.log("[saveTrainingData] Autosave disabled (set ALLOW_AUTOSAVE=true to enable). Skipping write to training_data.json");
    return;
  }

  trainingData.lastTrainingDate = new Date().toISOString();

  // Write atomically with a timestamped backup of the previous file.
  try {
    const tmpFile = `${trainingDataFile}.tmp`;
    const backupFile = `${trainingDataFile}.${Date.now()}.bak`;
    if (fs.existsSync(trainingDataFile)) {
      // Keep a backup of the previous state before overwriting.
      fs.copyFileSync(trainingDataFile, backupFile);
    }
    fs.writeFileSync(tmpFile, JSON.stringify(trainingData, null, 2));
    fs.renameSync(tmpFile, trainingDataFile);
    console.log(`[saveTrainingData] training_data.json saved (backup: ${backupFile})`);
  } catch (err) {
    console.error("[saveTrainingData] Failed to save training data:", err);
  }
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

// Helper function to detect if input is gibberish/random characters
function isGibberish(text) {
  if (!text || text.length < 3) return false;
  
  // Check for excessive repeated characters
  const repeatedChars = text.match(/(.)\1{3,}/g);
  if (repeatedChars && repeatedChars.length > 0) return true;
  
  // Check for lack of vowels (most real words have vowels)
  const vowelCount = (text.match(/[aeiou]/gi) || []).length;
  const vowelRatio = vowelCount / text.length;
  if (vowelRatio < 0.15 && text.length > 5) return true;
  
  // Check for random consonant clusters
  const consonantClusters = text.match(/[bcdfghjklmnpqrstvwxyz]{5,}/gi);
  if (consonantClusters && consonantClusters.length > 0) return true;
  
  // Check character diversity (gibberish often has too many unique chars in short span)
  const uniqueChars = new Set(text.toLowerCase()).size;
  if (text.length > 10 && uniqueChars > text.length * 0.8) return true;
  
  return false;
}

// Summarize Wikipedia content using AI
async function summarizeWikipediaContent(wikiText, topic) {
  if (!wikiText || wikiText.length < 100) return wikiText;
  
  // If content is already short, return as-is
  if (wikiText.length < 500) return wikiText;
  
  // Extract first 3-4 sentences as a natural summary
  const sentences = wikiText.split(/[.!?]+/).filter(s => s.trim().length > 20);
  
  // Take first 3 sentences, or up to ~300 characters
  let summary = [];
  let totalLength = 0;
  
  for (let i = 0; i < sentences.length && i < 4; i++) {
    const sentence = sentences[i].trim();
    if (totalLength + sentence.length > 400 && summary.length > 0) break;
    summary.push(sentence);
    totalLength += sentence.length;
  }
  
  const result = summary.join('. ').trim() + '.';
  
  // Add a note that it's summarized
  return `${result}\n\n[Summarized from Wikipedia - ${Math.round((result.length / wikiText.length) * 100)}% of original length]`;
}

// Wikipedia info fetching with better search relevance
async function getWikipediaInfo(query, previousContext = null) {
  // Clean the query more aggressively to get better search results
  const sanitizedQuery = query
    .toLowerCase()
    .replace(/^(what is|what are|who is|who are|describe|explain|tell me about|when did|where is|how did)\s+/i, '')
    .replace(/[?.,!]/g, '')
    .trim();

  // Don't use previous context for proper nouns or specific people/things
  const isProperNoun = /^[A-Z]/.test(query.trim());
  const useContext = !isProperNoun && previousContext;

  try {
    let searchQuery = sanitizedQuery;
    
    // Only add context if it's relevant and not a proper noun
    if (useContext) {
      const contextWords = previousContext.split(' ')
        .filter(word => word.length > 3)
        .slice(-2) // Reduced from 3 to 2 to avoid confusion
        .join(' ');
      searchQuery = `${sanitizedQuery} ${contextWords}`;
    }

    console.log(`[Wikipedia] Searching for: "${searchQuery}"`);
    const searchResults = await wiki().search(searchQuery);
    
    if (!searchResults.results || !searchResults.results.length) {
      console.log(`[Wikipedia] No results found for "${searchQuery}"`);
      return `Sorry, I couldn't find any relevant information about "${query}".`;
    }

    console.log(`[Wikipedia] Found: ${searchResults.results[0]}`);
    
    // Verify the result is actually relevant to the query
    const firstResult = searchResults.results[0].toLowerCase();
    const queryTerms = sanitizedQuery.toLowerCase().split(' ');
    const relevanceScore = queryTerms.filter(term => 
      term.length > 3 && firstResult.includes(term)
    ).length;

    // If relevance is too low, the result is probably wrong
    if (relevanceScore === 0 && queryTerms.length > 1) {
      console.log(`[Wikipedia] Result "${searchResults.results[0]}" doesn't seem relevant to "${sanitizedQuery}"`);
      return `Sorry, I couldn't find relevant information about "${query}". The search returned unrelated results.`;
    }

    const page = await wiki().page(searchResults.results[0]);
    const [summary, references] = await Promise.all([
      page.summary(),
      page.references().catch(() => [])
    ]);

    // Summarize the Wikipedia content
    let response = await summarizeWikipediaContent(summary, searchQuery);

    if (references && references.length > 0) {
      response += `\nSource: ${references[0]}`;
    }

    return response;
  } catch (error) {
    console.error(`[Wikipedia] Error fetching data for "${sanitizedQuery}":`, error);
    return `Sorry, I couldn't find any relevant information about "${query}".`;
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

  // Check for gibberish input
  if (isGibberish(message)) {
    console.log(`[Chat] Gibberish detected: "${message}"`);
    return res.json({
      response: "I can't understand that. Could you please type something more clear?",
      html: "<div class='ai-response'>I can't understand that. Could you please type something more clear?</div>"
    });
  }

  try {
    // Initialize conversation data for this chat if it doesn't exist
    if (!conversationData.has(chatId)) {
      conversationData.set(chatId, []);
    }

    // Get chat history and filter to ensure integrity
    let chatHistory = conversationData.get(chatId) || [];
    
    // SAFETY CHECK: Remove any corrupted or malformed messages
    chatHistory = chatHistory.filter(msg => 
      msg && 
      msg.sender && 
      msg.text && 
      (msg.sender === 'User' || msg.sender === 'AI') &&
      typeof msg.text === 'string' &&
      msg.text.length > 0
    );

    // Prevent AI from responding to itself by checking last message
    if (chatHistory.length > 0) {
      const lastMsg = chatHistory[chatHistory.length - 1];
      if (lastMsg.sender === 'User' && lastMsg.text === message) {
        console.log('[Chat] Duplicate user message detected, skipping...');
        return res.json({ 
          response: "I already received that message.", 
          html: "<div class='system-message'>Message already received.</div>" 
        });
      }
    }

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

    // 3. Wiki/Info questions - now handles contextual questions better
    } else if (
      messageForChecks.includes("wiki") ||
      messageForChecks.match(/^(what (is|are|does|did|was|were)|who (is|was|are|were)|explain|describe|when did|where is|how did)/i)
    ) {
      // Use resolved input for better context
      const resolvedMessage = responseGenerator.resolveReferences ? 
        responseGenerator.resolveReferences(message, chatHistory) : 
        message;

      cleanedMessage = resolvedMessage
        .replace(/^(wiki|what is|who is|what are|describe|explain|explain to me|when did|where is|how did|what does)\s*/i, '')
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
      const possibilities = await responseGenerator.generateEnhancedResponse(resolvedMessage, chatHistory);
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

    // Save conversation history - FIXED: Only save user's actual message
    chatHistory.push({ 
      sender: 'User', 
      text: message,
      timestamp: new Date().toISOString()
    });
    
    if (response) {
      chatHistory.push({ 
        sender: 'AI', 
        text: response,
        timestamp: new Date().toISOString()
      });
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
