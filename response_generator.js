/*
  response_generator.js
*/

// Configurable: disable Bing search if needed
const DISABLE_BING = true; // Set to true to disable Bing API
const TemplateMatcher = require("./template_matcher");
const tf = require("@tensorflow/tfjs-node-gpu");
const fs = require("fs");
const path = require("path");
const axios = require("axios");
const levenshtein = require("fast-levenshtein");
const readline = require("readline");
const chalk = require("chalk");
const natural = require("natural");
const tokenizer = new natural.WordTokenizer();

// Global cache for model and responses
const globalCache = {
  model: null,
  responses: new Map(),
  embeddings: new Map(),
  lastUpdate: "2025-02-05 04:38:59",
};

class ResponseGenerator {
  constructor(
    knowledgePath = "knowledge.json",
    trainingDataPath = "training_data.json",
    modelPath = "model/"
  ) {
    this.matcher = new TemplateMatcher(knowledgePath, trainingDataPath);
    // Convert to absolute path
    this.modelPath = path.resolve(modelPath);
    this.vocab = {};
    this.trainingDataPath = trainingDataPath;
    this.trainingData = {
      conversations: [],
      definitions: [],
      vocabulary: {},
      lastTrainingDate: "2025-02-05 04:38:59",
    };
    this.currentUser = "GMM-rgb";
    this.currentDateTime = "2025-02-05 04:38:59";
    this.modelCache = new Map();
    this.responseCache = new Map();

    // NEW: Question-answer tracking
    this.lastAIQuestion = null;
    this.waitingForAnswer = false;

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
    this.modelCache.maxSize = 2500;
    this.responseCache.maxSize = 500;

    setInterval(() => {
      const now = new Date().getTime();
      for (const [key, value] of this.modelCache) {
        if (now - value.timestamp > 3600000) {
          this.modelCache.delete(key);
        }
      }
    }, 900000);
  }

  async loadModel() {
    if (globalCache.model) {
      console.log(chalk.green("✅ Using cached model"));
      this.model = globalCache.model;
      return;
    }

    try {
      // Ensure model directory exists
      if (!fs.existsSync(this.modelPath)) {
        fs.mkdirSync(this.modelPath, { recursive: true });
      }

      // Check if model.json exists (not just if directory has files)
      const modelJsonPath = path.join(this.modelPath, "model.json");
      
      if (fs.existsSync(modelJsonPath)) {
        console.log(chalk.yellow("📂 Found existing model, loading..."));
        this.model = await tf.loadLayersModel(
          `file://${modelJsonPath}`
        );
        globalCache.model = this.model;
        console.log(chalk.green("✅ Existing model loaded successfully!"));
        console.log(chalk.cyan(`   Model has ${this.model.layers.length} layers`));
        return;
      }

      console.log(chalk.yellow("⚠️  No existing model found. Creating new model..."));
      console.log(chalk.yellow("   This is normal on first run."));
      this.model = await this.createNewModel();
      globalCache.model = this.model;
      
      // Save immediately after creation
      console.log(chalk.cyan("💾 Saving newly created model..."));
      const savePath = `file://${this.modelPath.replace(/\\/g, '/')}`;
      try {
        await this.model.save(savePath);
        console.log(chalk.green("✅ New model created and saved to disk"));
        console.log(chalk.cyan(`   Location: ${this.modelPath}`));
      } catch (saveError) {
        if (saveError && saveError.message && saveError.message.includes('save handlers')) {
          console.error(chalk.red("❌ Error saving model: Multiple tfjs-node packages detected. Please uninstall either @tensorflow/tfjs-node or @tensorflow/tfjs-node-gpu."));
        } else {
          console.error(chalk.red("❌ Error saving model:"), saveError);
        }
      }
    } catch (error) {
      console.error(chalk.red("❌ Error loading model:"), error);
      console.log(chalk.yellow("⚠️  Creating emergency backup model..."));
      this.model = await this.createNewModel();
      globalCache.model = this.model;
      
      // Try to save the backup model
      try {
        await this.model.save(`file://${this.modelPath}`);
        console.log(chalk.green("✅ Backup model saved"));
      } catch (saveError) {
        console.error(chalk.red("❌ Could not save backup model:"), saveError);
      }
    }
  }

  async createNewModel() {
    const model = tf.sequential();

    model.add(
      tf.layers.embedding({
        inputDim: 10000,
        outputDim: 128,
        inputLength: 50,
      })
    );

    model.add(
      tf.layers.lstm({
        units: 64,
        returnSequences: true,
      })
    );

    model.add(
      tf.layers.lstm({
        units: 32,
      })
    );

    model.add(
      tf.layers.dense({
        units: 64,
        activation: "relu",
      })
    );

    model.add(
      tf.layers.dense({
        units: 32,
        activation: "relu",
      })
    );

    model.add(
      tf.layers.dense({
        units: 16,
        activation: "softmax",
      })
    );

    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });
    console.log(chalk.green("✅ New model created"));
    return model;
  }

  loadTrainingData() {
    if (fs.existsSync(this.trainingDataPath)) {
      try {
        const data = fs.readFileSync(this.trainingDataPath, "utf8");
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
      lastTrainingDate: this.currentDateTime,
    };
  }

  initializeTFIDF() {
    this.tfidf = new natural.TfIdf();
    this.trainingData.conversations.forEach((conv) => {
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
      user: this.currentUser,
    });
    await this.saveTrainingData();
  }

  async saveTrainingData() {
    // Respect environment flag to prevent uncontrolled overwrites.
    if (process.env.ALLOW_AUTOSAVE !== "true") {
      console.log("[ResponseGenerator.saveTrainingData] Autosave disabled (set ALLOW_AUTOSAVE=true to enable). Skipping write to training_data.json");
      return;
    }

    this.trainingData.lastTrainingDate = new Date().toISOString();

    // Write atomically with a timestamped backup of the previous file.
    try {
      const tmpFile = `${this.trainingDataPath}.tmp`;
      const backupFile = `${this.trainingDataPath}.${Date.now()}.bak`;
      if (fs.existsSync(this.trainingDataPath)) {
        fs.copyFileSync(this.trainingDataPath, backupFile);
      }
      fs.writeFileSync(tmpFile, JSON.stringify(this.trainingData, null, 2));
      fs.renameSync(tmpFile, this.trainingDataPath);
      console.log(`[ResponseGenerator.saveTrainingData] training_data.json saved (backup: ${backupFile})`);
    } catch (err) {
      console.error("[ResponseGenerator.saveTrainingData] Failed to save training data:", err);
    }
  }

  // NEW: Resolve references like "this", "that", "it" to actual content
  // FIXED: Only look at USER messages, not AI responses
  resolveReferences(inputText, chatHistory) {
    if (!chatHistory || chatHistory.length === 0) return inputText;

    const referenceWords = ['this', 'that', 'it', 'these', 'those', 'them'];
    const hasReference = referenceWords.some(ref => 
      inputText.toLowerCase().includes(ref)
    );

    if (!hasReference) return inputText;

    // Find the most recent relevant content FROM USER ONLY
    const recentMessages = chatHistory
      .slice(-5)
      .filter(msg => msg.sender === 'User') // ONLY look at user messages
      .reverse();
    
    for (const msg of recentMessages) {
      // Skip very short messages
      if (msg.text.length < 10) continue;
      
      // Extract key content from user's previous messages
      const match = msg.text.match(/^(.*?)[.!?]/);
      if (match) {
        const firstSentence = match[1].trim();
        // Replace reference words with actual content
        let resolved = inputText;
        referenceWords.forEach(ref => {
          const pattern = new RegExp(`\\b${ref}\\b`, 'gi');
          resolved = resolved.replace(pattern, `"${firstSentence}"`);
        });
        console.log(`[Context] Resolved "${inputText}" to "${resolved}"`);
        return resolved;
      }
    }

    return inputText;
  }

  // NEW: Detect if user is answering AI's previous question
  detectIfAnsweringPreviousQuestion(input, chatHistory) {
    if (!chatHistory || chatHistory.length < 2) return null;

    const lastAIMessage = chatHistory
      .slice()
      .reverse()
      .find((msg) => msg.sender === "AI");

    if (!lastAIMessage || !lastAIMessage.text.includes("?")) {
      return null;
    }

    const question = lastAIMessage.text;
    const questionKeywords = this.extractKeyTerms(question);
    const answerKeywords = this.extractKeyTerms(input);

    const relevance =
      questionKeywords.filter((kw) => answerKeywords.includes(kw)).length /
      Math.max(questionKeywords.length, 1);

    const intent = this.detectUserIntent(input, chatHistory);
    const isDirectAnswer =
      intent.type === "CONFIRMATION" ||
      intent.type === "NEGATION" ||
      relevance > 0.2;

    return {
      question: question,
      isAnswer: isDirectAnswer,
      relevance: relevance,
    };
  }

  // NEW: Generate acknowledgment for answers
  generateAcknowledgment(answer, previousQuestion) {
    const intent = this.detectUserIntent(answer);
    const questionTopic = this.extractKeyTerms(previousQuestion)[0] || "that";

    const templates = {
      CONFIRMATION: [
        "I understand. ",
        "Got it. ",
        "Okay. ",
        "That makes sense. ",
      ],
      NEGATION: [
        "I see. ",
        "Understood. ",
        "Alright. ",
        "Fair enough. ",
      ],
      default: [
        "Thanks for letting me know. ",
        "I appreciate that answer. ",
        "That helps clarify things. ",
        "Good to know. ",
      ],
    };

    const template = templates[intent.type] || templates["default"];
    const prefix = template[Math.floor(Math.random() * template.length)];

    return `${prefix}${this.generateFollowUp(answer, questionTopic)}`;
  }

  // NEW: Generate contextual follow-up
  generateFollowUp(answer, topic) {
    const sentiment = this.analyzeSentiment(answer);

    if (sentiment.label === "positive") {
      return `That's great! Is there anything else you'd like to know about ${topic}?`;
    } else if (sentiment.label === "negative") {
      return `I understand. Would you like to discuss something else?`;
    } else {
      return `Would you like me to elaborate on ${topic}?`;
    }
  }

  // UPDATED: Enhanced response generation with question-answer awareness
  async generateEnhancedResponse(inputText, chatHistory = []) {
    if (!inputText) return null;

    // First, resolve any references to previous content
    const resolvedInput = this.resolveReferences(inputText, chatHistory);

    const possibilities = [];
    const context = this.buildResponseContext(resolvedInput, chatHistory);
    const understanding = await this.analyzeContext(resolvedInput, chatHistory);

    // Check if user is answering AI's previous question
    const questionContext = this.detectIfAnsweringPreviousQuestion(
      resolvedInput,
      chatHistory
    );

    if (questionContext && questionContext.isAnswer) {
      const acknowledgment = this.generateAcknowledgment(
        resolvedInput,
        questionContext.question
      );
      possibilities.push({
        response: acknowledgment,
        confidence: 0.95,
        source: "question_answer_pair",
      });
    }

    const conversationThread = this.findConversationThread(
      resolvedInput,
      chatHistory
    );
    const contextBoost = conversationThread ? 1.2 : 1;

    // Direct match with improved confidence calculation - BUT STRICTER
    const directMatch = this.findClosestMatch(resolvedInput, context);
    if (directMatch && directMatch.confidence > 0.75) { // Increased from 0.5 to 0.75
      // Additional check: make sure the match actually makes sense
      const inputWords = resolvedInput.toLowerCase().split(' ').filter(w => w.length > 3);
      const matchWords = directMatch.output.toLowerCase().split(' ').filter(w => w.length > 3);
      const overlap = inputWords.filter(w => matchWords.includes(w)).length;
      
      // Only use if there's actual topic overlap
      if (overlap > 0 || directMatch.confidence > 0.9) {
        possibilities.push({
          response: this.properlyCapitalize(
            this.addContextToResponse(directMatch.output, understanding)
          ),
          confidence: directMatch.confidence * contextBoost,
          source: "direct_match",
        });
      }
    }

    // Template matcher - also stricter
    const templateMatch = this.matcher.findBestTemplate(resolvedInput);
    if (templateMatch.bestMatch && templateMatch.confidence > 0.65) { // Increased from 0.5
      possibilities.push({
        response: this.properlyCapitalize(templateMatch.bestMatch.output),
        confidence: templateMatch.confidence * contextBoost,
        source: "template_match",
      });
    }

    // Similar responses with context
    const similarResponses = await this.findSimilarResponsesWithContext(
      resolvedInput,
      context,
      understanding
    );
    
    // Filter out responses with random irrelevant content
    const filteredSimilar = similarResponses.filter(resp => {
      const respLower = resp.response.toLowerCase();
      // Don't return responses about geology, math errors, etc. unless asked
      const irrelevantTopics = ['law of superposition', 'rock strata', 'geology', 'couldn\'t solve that math'];
      return !irrelevantTopics.some(topic => respLower.includes(topic));
    });
    
    possibilities.push(...filteredSimilar);

    // Model response only if no good matches
    if (possibilities.length === 0 || possibilities[0].confidence < 0.7) {
      const modelResponse = await this.generateModelResponseWithContext(
        resolvedInput,
        understanding,
        chatHistory
      );
      if (modelResponse) {
        possibilities.push({
          response: this.properlyCapitalize(modelResponse),
          confidence: 0.6 * contextBoost,
          source: "ai_model",
        });
      }
    }

    // Sort by confidence
    const sortedResponses = possibilities
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 3);

    // Learn from the interaction
    if (sortedResponses.length > 0) {
      await this.learnFromInteraction(
        resolvedInput,
        sortedResponses[0].response,
        understanding
      );
    }

    // Return best responses or fallback
    return sortedResponses.length > 0
      ? sortedResponses
      : [
          {
            response:
              "I'm not quite sure how to respond to that. Could you rephrase or provide more context?",
            confidence: 0.3,
            source: "fallback",
          },
        ];
  }

  // UPDATED: Fixed similarity calculation in findClosestMatch
  findClosestMatch(inputText, context) {
    if (!inputText || !this.trainingData.conversations) return null;

    const normalizedInput = inputText.toLowerCase().trim();
    let bestMatch = null;
    let bestSimilarity = 0;

    this.trainingData.conversations.forEach((conv) => {
      if (!conv || !conv.input) return;

      const distance = levenshtein.get(
        normalizedInput,
        conv.input.toLowerCase().trim()
      );

      const maxLen = Math.max(normalizedInput.length, conv.input.length);
      const similarity = 1 - distance / maxLen;

      if (similarity > bestSimilarity && similarity > 0.4) {
        bestSimilarity = similarity;
        bestMatch = {
          output: conv.output,
          confidence: similarity,
        };
      }
    });

    return bestMatch;
  }

  async analyzeContext(input, chatHistory) {
    const understanding = {
      topic: await this.detectTopic(input),
      references: await this.findReferences(input),
      sentiment: this.analyzeSentiment(input),
      previousContext: this.extractPreviousContext(chatHistory),
      userIntent: this.detectUserIntent(input, chatHistory),
    };

    return understanding;
  }

  async detectTopic(input) {
    const keyTerms = this.extractKeyTerms(input);
    const relatedTopics = await Promise.all(
      keyTerms.map((term) => this.searchKnowledgeBase(term))
    );

    return {
      mainTopic: keyTerms[0],
      relatedTopics: relatedTopics.filter(Boolean),
    };
  }

  async findReferences(input) {
    if (!input) return [];

    try {
      const references = [];
      const keyTerms = this.extractKeyTerms(input);

      for (const term of keyTerms) {
        const matches = this.trainingData.conversations.filter(
          (conv) =>
            conv.input.toLowerCase().includes(term.toLowerCase()) ||
            conv.output.toLowerCase().includes(term.toLowerCase())
        );

        matches.forEach((match) => {
          if (match.source) {
            references.push({
              term,
              source: match.source,
              confidence: this.calculateSimilarity(input, match.input),
            });
          }
        });
      }

      if (references.length === 0) {
        for (const term of keyTerms.slice(0, 2)) {
          const wikiInfo = await this.searchWikipedia(term);
          if (wikiInfo) {
            references.push({
              term,
              source: "Wikipedia",
              confidence: 0.7,
            });
          }
        }
      }

      return Array.from(
        new Set(
          references
            .sort((a, b) => b.confidence - a.confidence)
            .map((ref) => ref.source)
        )
      );
    } catch (error) {
      console.error("Error finding references:", error);
      return [];
    }
  }

  calculateSimilarity(str1, str2) {
    if (!str1 || !str2) return 0;

    const set1 = new Set(str1.toLowerCase().split(" "));
    const set2 = new Set(str2.toLowerCase().split(" "));

    const intersection = new Set([...set1].filter((x) => set2.has(x)));
    const union = new Set([...set1, ...set2]);

    return intersection.size / union.size;
  }

  findConversationThread(input, history) {
    if (!history || history.length === 0) return null;

    const continuityMarkers = [
      "that",
      "it",
      "this",
      "those",
      "these",
      "they",
      "the",
      "your",
      "my",
      "our",
      "their",
    ];

    const hasMarkers = continuityMarkers.some((marker) =>
      input.toLowerCase().includes(marker)
    );

    if (hasMarkers) {
      const relevantMessage = history
        .slice()
        .reverse()
        .find((msg) => {
          const similarity = this.calculateSimilarity(input, msg.text);
          return similarity > 0.3;
        });

      if (relevantMessage) {
        return {
          previousMessage: relevantMessage,
          continuityScore: this.calculateContinuityScore(input, relevantMessage),
        };
      }
    }

    return null;
  }

  addContextToResponse(response, understanding) {
    if (!understanding || !response) return response;

    if (understanding.previousContext) {
      response = this.addPreviousContext(response, understanding.previousContext);
    }

    if (understanding.references && understanding.references.length > 0) {
      response += "\n\nSources: " + understanding.references.join(", ");
    }

    return response;
  }

  async generateModelResponseWithContext(input, understanding, history) {
    const contextInput = this.prepareContextInput(input, understanding, history);
    let response = await this.generateModelResponse(contextInput);

    if (
      this.detectUserIntent(input).type === "QUESTION" ||
      input.toLowerCase().includes("what") ||
      input.toLowerCase().includes("how") ||
      input.toLowerCase().includes("why")
    ) {
      response = await this.addWebReferences(response, input);
    }

    return this.addContextToResponse(response, understanding);
  }

  prepareContextInput(input, understanding, history) {
    let contextInput = input;

    if (history && history.length > 0) {
      const relevantHistory = history
        .slice(-3)
        .map((msg) => `${msg.sender}: ${msg.text}`)
        .join("\n");
      contextInput = `Previous messages:\n${relevantHistory}\n\nCurrent message: ${input}`;
    }

    if (understanding.topic) {
      contextInput += `\nContext: ${understanding.topic.mainTopic}`;
    }

    return contextInput;
  }

  buildResponseContext(inputText, chatHistory) {
    return {
      recentConversations: this.trainingData.conversations.slice(-5),
      currentTime: this.currentDateTime,
      currentUser: this.currentUser,
      vocabulary: this.vocab,
      chatHistory: chatHistory,
    };
  }

  async findSimilarResponsesWithContext(inputText, context, understanding) {
    if (!inputText || !this.trainingData.conversations) return [];

    const responses = [];
    const topicKeywords = understanding.topic?.mainTopic
      ? [understanding.topic.mainTopic]
      : [];
    const sentimentScore = understanding.sentiment?.score || 0;

    const baseMatches = this.findSimilarResponses(inputText, context);

    for (const match of baseMatches) {
      let contextualConfidence = match.similarity;

      if (topicKeywords.length > 0) {
        const matchTopics = this.extractKeyTerms(match.input);
        const topicOverlap = topicKeywords.filter((topic) =>
          matchTopics.includes(topic)
        ).length;
        contextualConfidence *= 1 + topicOverlap * 0.25;
      }

      const matchSentiment = this.analyzeSentiment(match.output).score;
      const sentimentAlignment = 1 - Math.abs(sentimentScore - matchSentiment) / 2;
      contextualConfidence *= sentimentAlignment;

      responses.push({
        response: this.properlyCapitalize(match.output),
        confidence: contextualConfidence,
        source: "contextual_match",
      });
    }

    return responses.sort((a, b) => b.confidence - a.confidence).slice(0, 5);
  }

  findSimilarResponses(inputText, context) {
    if (!inputText || !this.trainingData.conversations) return [];

    const inputTokens = this.tokenizer.tokenize(inputText.toLowerCase());
    const inputIntent = this.detectUserIntent(inputText);

    return this.trainingData.conversations
      .map((conv) => {
        if (!conv || !conv.input || !conv.output) return null;

        const convTokens = this.tokenizer.tokenize(conv.input.toLowerCase());
        const commonTokens = inputTokens.filter((token) =>
          convTokens.includes(token)
        );

        const tfidfSimilarity = this.calculateTFIDFSimilarity(
          inputText,
          conv.input
        );
        const tokenSimilarity =
          commonTokens.length / Math.max(inputTokens.length, convTokens.length);

        let similarity = tfidfSimilarity * 0.7 + tokenSimilarity * 0.3;

        // PENALTY: If the output seems completely unrelated to input, reduce similarity
        const outputTokens = this.tokenizer.tokenize(conv.output.toLowerCase());
        const outputRelevance = inputTokens.filter(t => outputTokens.includes(t)).length;
        if (outputRelevance === 0 && similarity < 0.7) {
          similarity *= 0.5; // Heavy penalty for unrelated outputs
        }

        // PENALTY: If intents don't match at all, reduce similarity
        const convIntent = this.detectUserIntent(conv.input);
        if (inputIntent.type !== convIntent.type && similarity < 0.8) {
          similarity *= 0.8;
        }

        return {
          ...conv,
          similarity,
        };
      })
      .filter((conv) => conv && conv.similarity > 0.4) // Increased from 0.3
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
      ...this.tokenizer.tokenize(text2.toLowerCase()),
    ]);

    terms.forEach((term) => {
      const score1 = tfidf.tfidf(term, 0);
      const score2 = tfidf.tfidf(term, 1);
      similarity += Math.min(score1, score2);
    });

    return similarity / terms.size;
  }

  extractKeyTerms(text) {
    if (!text) return [];

    const cleanText = text.toLowerCase().replace(/[^\w\s]/g, "");
    const tokens = this.tokenizer.tokenize(cleanText);

    const stopwords = new Set([
      "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
      "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
      "to", "was", "were", "will", "with",
    ]);

    const filteredTokens = tokens.filter((token) => !stopwords.has(token));

    const termFreq = {};
    filteredTokens.forEach((token) => {
      termFreq[token] = (termFreq[token] || 0) + 1;
    });

    const sortedTerms = Object.entries(termFreq)
      .sort(([, a], [, b]) => b - a)
      .map(([term]) => term);

    return sortedTerms.slice(0, 5);
  }

  async searchKnowledgeBase(term) {
    if (!term) return null;

    try {
      const cacheKey = `kb_${term.toLowerCase()}`;
      if (this.modelCache.has(cacheKey)) {
        return this.modelCache.get(cacheKey).data;
      }

      const relevantData = this.trainingData.conversations.find(
        (conv) =>
          conv.input.toLowerCase().includes(term.toLowerCase()) ||
          conv.output.toLowerCase().includes(term.toLowerCase())
      );

      if (relevantData) {
        this.modelCache.set(cacheKey, {
          data: relevantData.output,
          timestamp: Date.now(),
        });
        return relevantData.output;
      }

      const wikiResult = await this.searchWikipedia(term);
      if (wikiResult) {
        this.modelCache.set(cacheKey, {
          data: wikiResult,
          timestamp: Date.now(),
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
      const wiki = require("wikijs").default;
      
      // Clean the search term
      const cleanTerm = term.trim();
      console.log(`[Wikipedia Search] Looking for: "${cleanTerm}"`);
      
      const searchResults = await wiki().search(cleanTerm);
      
      if (!searchResults.results || searchResults.results.length === 0) {
        console.log(`[Wikipedia Search] No results found for "${cleanTerm}"`);
        return null;
      }

      const firstResult = searchResults.results[0];
      console.log(`[Wikipedia Search] First result: "${firstResult}"`);
      
      // Verify relevance: check if search term appears in result title
      const termWords = cleanTerm.toLowerCase().split(' ').filter(w => w.length > 2);
      const resultLower = firstResult.toLowerCase();
      const matchCount = termWords.filter(word => resultLower.includes(word)).length;
      
      // If less than 30% of words match, result is probably irrelevant
      if (matchCount < termWords.length * 0.3) {
        console.log(`[Wikipedia Search] Result "${firstResult}" seems irrelevant to "${cleanTerm}" (only ${matchCount}/${termWords.length} words match)`);
        return null;
      }

      const page = await wiki().page(firstResult);
      const summary = await page.summary();
      
      // Summarize if too long (over 500 chars)
      let finalSummary = summary;
      if (summary.length > 500) {
        const sentences = summary.split(/[.!?]+/).filter(s => s.trim().length > 20);
        const briefSummary = [];
        let totalLength = 0;
        
        for (let i = 0; i < sentences.length && i < 3; i++) {
          const sentence = sentences[i].trim();
          if (totalLength + sentence.length > 400 && briefSummary.length > 0) break;
          briefSummary.push(sentence);
          totalLength += sentence.length;
        }
        
        finalSummary = briefSummary.join('. ').trim() + '.';
        finalSummary += `\n\n[Summarized from Wikipedia]`;
      }
      
      console.log(`[Wikipedia Search] Successfully retrieved summary for "${firstResult}"`);
      return finalSummary;
    } catch (error) {
      console.error(`[Wikipedia Search] Error searching for term "${term}":`, error);
      return null;
    }
  }

  analyzeSentiment(text) {
    if (!text) return { score: 0, label: "neutral" };

    const positiveWords = new Set([
      "good", "great", "awesome", "excellent", "happy", "love",
      "wonderful", "fantastic", "amazing", "thanks", "yes", "yeah",
    ]);

    const negativeWords = new Set([
      "bad", "terrible", "awful", "horrible", "sad", "hate",
      "poor", "worst", "annoying", "sorry", "no", "nope",
    ]);

    const words = text.toLowerCase().split(/\s+/);
    let score = 0;

    words.forEach((word) => {
      if (positiveWords.has(word)) score += 1;
      if (negativeWords.has(word)) score -= 1;
    });

    return {
      score,
      label: score > 0 ? "positive" : score < 0 ? "negative" : "neutral",
    };
  }

  extractPreviousContext(chatHistory) {
    if (!chatHistory || chatHistory.length === 0) return null;

    const recentMessages = chatHistory.slice(-3);

    return {
      lastMessage: recentMessages[recentMessages.length - 1],
      recentContext: recentMessages.map((msg) => ({
        role: msg.sender.toLowerCase(),
        content: msg.text,
      })),
      topics: this.extractKeyTerms(
        recentMessages.map((msg) => msg.text).join(" ")
      ),
    };
  }

  // UPDATED: Detect intent with better contextual understanding
  detectUserIntent(input, chatHistory = []) {
    const normalizedInput = input.toLowerCase().trim();
    
    // Enhanced pattern matching that considers context
    const intents = {
      QUESTION_ABOUT_PREVIOUS: {
        pattern: /^(what (does|is|are|was|were)|explain|tell me about|describe) (this|that|it|these|those)/i,
        contextRequired: true
      },
      MATH_EVALUATION: {
        pattern: /^(what (does|is)|calculate|solve|compute) .*(equal|equals|\+|\-|\*|\/|\^)/i,
        contextRequired: false
      },
      DEFINITION_REQUEST: {
        pattern: /^(what (does|is|are|was|were)|define|meaning of|definition of)/i,
        contextRequired: false
      },
      QUESTION: /^(what|who|where|when|why|how|can|could|would|will|do|does|did|is|are|was|were)/i,
      GREETING: /^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))/i,
      FAREWELL: /^(bye|goodbye|see\s+you|farewell)/i,
      GRATITUDE: /(thank|thanks)/i,
      REQUEST: /^(please|can\s+you|could\s+you|would\s+you)/i,
      CONFIRMATION: /^(yes|yeah|yep|sure|okay|ok|alright|correct|right|exactly)/i,
      NEGATION: /^(no|nope|nah|not|never|wrong)/i,
    };

    // Check context-aware intents first
    for (const [intent, config] of Object.entries(intents)) {
      if (typeof config === 'object' && config.pattern) {
        if (config.pattern.test(normalizedInput)) {
          if (config.contextRequired && chatHistory && chatHistory.length > 0) {
            return {
              type: intent,
              confidence: 0.9,
              metadata: {
                requiresContext: true,
                pattern: config.pattern.source,
                match: input.match(config.pattern)[0],
              },
            };
          } else if (!config.contextRequired) {
            return {
              type: intent,
              confidence: 0.85,
              metadata: {
                pattern: config.pattern.source,
                match: input.match(config.pattern)[0],
              },
            };
          }
        }
      } else if (config.test && config.test(normalizedInput)) {
        return {
          type: intent,
          confidence: 0.8,
          metadata: {
            pattern: config.source,
            match: input.match(config)[0],
          },
        };
      }
    }

    // Check if responding to AI's previous question
    if (chatHistory && chatHistory.length > 0) {
      const lastMessage = chatHistory[chatHistory.length - 1];
      if (lastMessage.sender === "AI" && lastMessage.text.endsWith("?")) {
        return {
          type: "RESPONSE_TO_QUESTION",
          confidence: 0.6,
          metadata: {
            previousQuestion: lastMessage.text,
          },
        };
      }
    }

    return {
      type: "STATEMENT",
      confidence: 0.5,
      metadata: {},
    };
  }

  calculateContinuityScore(input, previousMessage) {
    if (!input || !previousMessage || !previousMessage.text) return 0;

    const baseSimilarity = this.calculateSimilarity(input, previousMessage.text);
    const timeDecay = previousMessage.timestamp
      ? Math.exp(
          -(Date.now() - new Date(previousMessage.timestamp).getTime()) /
            (1000 * 60 * 60)
        )
      : 1;

    return baseSimilarity * timeDecay;
  }

  addPreviousContext(response, context) {
    if (!context || !context.lastMessage) return response;

    const contextReferences = {
      it: context.lastMessage.text,
      that: context.lastMessage.text,
      this: context.lastMessage.text,
    };

    Object.entries(contextReferences).forEach(([pronoun, reference]) => {
      const regex = new RegExp(`\\b${pronoun}\\b`, "gi");
      if (response.match(regex)) {
        response = response.replace(regex, `"${reference}"`);
      }
    });

    return response;
  }

  properlyCapitalize(text) {
    if (!text) return text;

    const sentences = this.sentenceTokenizer.tokenize(text);
    return sentences
      .map((sentence) => {
        if (!sentence.trim()) return sentence;

        const specialWords = ["i", "i'm", "i'll", "i've", "i'd"];

        return sentence
          .split(" ")
          .map((word, index) => {
            if (index === 0 || specialWords.includes(word.toLowerCase())) {
              return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
            }
            return word.toLowerCase();
          })
          .join(" ");
      })
      .join(" ");
  }

  async generateModelResponse(inputText) {
    const cacheKey = inputText.toLowerCase().trim();
    if (this.responseCache.has(cacheKey)) {
      const cached = this.responseCache.get(cacheKey);
      if (new Date().getTime() - cached.timestamp < 3600000) {
        return cached.response;
      }
    }

    try {
      const tokens = this.tokenizer.tokenize(inputText.toLowerCase());
      const tokenIndices = tokens.map((token) => this.vocab[token] || 0);

      const paddedTokens = [
        ...tokenIndices.slice(0, 50),
        ...Array(Math.max(0, 50 - tokenIndices.length)).fill(0),
      ];

      const inputTensor = tf.tensor2d([paddedTokens], [1, 50]);
      const prediction = this.model.predict(inputTensor);

      let response;
      if (prediction.shape[1] === this.trainingData.conversations.length) {
        const responseIndex = tf.argMax(prediction, 1).dataSync()[0];
        const candidate = this.trainingData.conversations[responseIndex];
        
        // VALIDATION: Make sure the response makes sense for the input
        if (candidate && candidate.output) {
          const inputWords = inputText.toLowerCase().split(' ').filter(w => w.length > 3);
          const outputWords = candidate.output.toLowerCase().split(' ').filter(w => w.length > 3);
          const relevance = inputWords.filter(w => outputWords.includes(w)).length;
          
          // Only use if there's some relevance OR high confidence
          if (relevance > 0 || prediction.dataSync()[responseIndex] > 0.8) {
            response = candidate.output;
          }
        }
      }

      // Improved fallback order: TemplateMatcher, Wikipedia, KnowledgeBase, then Bing
      if (!response) {
        const { bestMatch } = this.matcher.findBestTemplate(inputText);
        if (bestMatch?.output) {
          response = bestMatch.output;
        } else {
          // Try Wikipedia
          const wiki = await this.searchWikipedia(inputText);
          if (wiki) {
            response = wiki;
          } else {
            // Try knowledge base
            const kb = await this.searchKnowledgeBase(inputText);
            if (kb) {
              response = kb;
            } else {
              // Try Bing (if enabled)
              const web = await this.fetchWebArticles(inputText);
              if (web && web.length > 0) {
                response = web[0].snippet || web[0].title;
              } else {
                response = "I'm still learning how to respond to that.";
              }
            }
          }
        }
      }

      this.responseCache.set(cacheKey, {
        response,
        timestamp: new Date().getTime(),
      });

      inputTensor.dispose();
      prediction.dispose();

      return response;
    } catch (error) {
      console.error(chalk.red("❌ Model response error:"), error);
      return "I encountered an error while processing your message.";
    }
  }

  async learnFromInteraction(input, output, understanding) {
    try {
      if (!input || !output) return;

      // Update training data first
      await this.updateTrainingData(input, output);
      
      // Only retrain model periodically (every 10 interactions) to avoid constant retraining
      const conversationCount = this.trainingData.conversations.length;
      const shouldRetrain = conversationCount % 10 === 0;

      if (shouldRetrain) {
        console.log(chalk.yellow(`📚 Learning checkpoint reached (${conversationCount} conversations)`));
        console.log(chalk.cyan("💾 Saving model with new knowledge..."));
        try {
          // Save the model to disk - always use forward slashes for Windows
          const savePath = `file://${this.modelPath.replace(/\\/g, '/')}`;
          try {
            await this.model.save(savePath);
            console.log(chalk.green("✅ Model saved successfully!"));
            console.log(chalk.cyan(`   Location: ${this.modelPath}`));
            // Update global cache timestamp
            globalCache.lastUpdate = new Date().toISOString();
          } catch (saveError) {
            if (saveError && saveError.message && saveError.message.includes('save handlers')) {
              console.error(chalk.red("❌ Error saving model: Multiple tfjs-node packages detected. Please uninstall either @tensorflow/tfjs-node or @tensorflow/tfjs-node-gpu."));
            } else {
              console.error(chalk.red("❌ Error saving model:"), saveError);
            }
          }
        } catch (saveOuterError) {
          console.error(chalk.red("❌ Error in save logic:"), saveOuterError);
        }
      } else {
        console.log(chalk.green(`✅ Learned from interaction (${conversationCount} total conversations)`));
      }
    } catch (error) {
      console.error(chalk.red("❌ Learning error:"), error);
    }
  }

  // Method to manually save the model
  async saveModel() {
    try {
      console.log(chalk.cyan("💾 Manually saving model..."));
      await this.model.save(`file://${this.modelPath}`);
      await this.saveTrainingData();
      console.log(chalk.green("✅ Model and training data saved!"));
      console.log(chalk.cyan(`   Location: ${this.modelPath}`));
      return true;
    } catch (error) {
      console.error(chalk.red("❌ Error saving model:"), error);
      return false;
    }
  }

  // Method to get model info
  getModelInfo() {
    return {
      layers: this.model.layers.length,
      trainableParams: this.model.countParams(),
      conversations: this.trainingData.conversations.length,
      vocabularySize: Object.keys(this.vocab).length,
      lastUpdate: globalCache.lastUpdate,
      modelPath: this.modelPath
    };
  }

  async addWebReferences(response, query) {
    const wikiResult = await this.searchWikipedia(query);
    const webArticles = await this.fetchWebArticles(query);

    let referencesText = "\n\n--- References ---\n";
    if (wikiResult) {
      referencesText += `Wikipedia: ${wikiResult}\n`;
    }
    if (webArticles && Array.isArray(webArticles) && webArticles.length) {
      webArticles.forEach((article) => {
        referencesText += `${article.title}: ${article.url}\n`;
      });
    }
    return response + referencesText;
  }

  async fetchWebArticles(query) {
    if (DISABLE_BING) {
      // Bing is disabled, skip web search
      return [];
    }
    try {
      const subscriptionKey = "1feda3372abf425494ce986ad9024238";
      if (!subscriptionKey || subscriptionKey === "" || subscriptionKey === "YOUR_BING_KEY") {
        console.warn("[INFO] Bing API key missing or disabled. Skipping Bing search.");
        return [];
      }
      const response = await axios({
        method: "get",
        url: "https://api.bing.microsoft.com/v7.0/search",
        headers: {
          "Ocp-Apim-Subscription-Key": subscriptionKey,
          Accept: "application/json",
        },
        params: {
          q: query,
          count: 3,
          responseFilter: "Webpages",
          mkt: "en-US",
        },
        timeout: 10000,
      });
      if (response.data?.webPages?.value) {
        return response.data.webPages.value.map((result) => ({
          url: result.url,
          title: result.name,
          snippet: result.snippet,
          source: new URL(result.url).hostname.replace(/^www\./, ""),
        }));
      }
      return [];
    } catch (error) {
      console.error("Error fetching Bing results:", error?.response?.data || error.message);
      return [];
    }
  }
}

module.exports = ResponseGenerator;
