const fs = require('fs');

function levenshteinDistance(s1, s2) {
    if (s1.length < s2.length) {
        return levenshteinDistance(s2, s1);
    }

    if (s2.length === 0) {
        return s1.length;
    }

    let previousRow = Array.from({ length: s2.length + 1 }, (_, i) => i);

    for (let i = 0; i < s1.length; i++) {
        const currentRow = [i + 1];
        for (let j = 0; j < s2.length; j++) {
            const insertions = previousRow[j + 1] + 1;
            const deletions = currentRow[j] + 1;
            const substitutions = previousRow[j] + (s1[i] !== s2[j] ? 1 : 0);
            currentRow.push(Math.min(insertions, deletions, substitutions));
        }
        previousRow = currentRow;
    }

    return previousRow[previousRow.length - 1];
}

class TemplateMatcher {
    constructor(knowledgePath, trainingDataPath) {
        this.knowledgePath = knowledgePath;
        this.trainingDataPath = trainingDataPath;
        this.knowledge = this._loadJson(knowledgePath);
        this.trainingData = this._loadJson(trainingDataPath);
        this.intentPatterns = this._buildIntentPatterns(); // Build patterns FIRST
        this.templates = this._extractTemplates(); // Then extract templates
    }

    _loadJson(path) {
        try {
            if (fs.existsSync(path)) {
                const data = fs.readFileSync(path, 'utf8');
                return JSON.parse(data);
            }
            return { conversations: [], vocabulary: {} };
        } catch (error) {
            console.error(`Error loading JSON from ${path}:`, error);
            return { conversations: [], vocabulary: {} };
        }
    }

    _extractTemplates() {
        const templates = [];
        
        // Extract from training data
        if (this.trainingData.conversations && Array.isArray(this.trainingData.conversations)) {
            for (const conversation of this.trainingData.conversations) {
                if (conversation.input && conversation.output) {
                    templates.push({
                        input: conversation.input.toLowerCase(),
                        output: conversation.output,
                        intent: this._detectIntent(conversation.input),
                        keywords: this._extractKeywords(conversation.input)
                    });
                }
            }
        }

        // Extract from knowledge base
        if (this.knowledge && typeof this.knowledge === 'object') {
            for (const [key, value] of Object.entries(this.knowledge)) {
                templates.push({
                    input: key.toLowerCase(),
                    output: value,
                    intent: this._detectIntent(key),
                    keywords: this._extractKeywords(key)
                });
            }
        }

        return templates;
    }

    _buildIntentPatterns() {
        return {
            GREETING: /^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))/i,
            FAREWELL: /^(bye|goodbye|see\s+you|farewell|talk\s+to\s+you\s+later)/i,
            QUESTION: /^(what|who|where|when|why|how|can|could|would|will|do|does|did|is|are|was|were)/i,
            GRATITUDE: /(thank|thanks|appreciate)/i,
            CONFIRMATION: /^(yes|yeah|yep|sure|okay|ok|alright|correct|right)/i,
            NEGATION: /^(no|nope|nah|not|never)/i,
        };
    }

    _detectIntent(text) {
        if (!text) return 'STATEMENT';
        
        const normalizedText = text.toLowerCase().trim();
        
        // Safety check - if intentPatterns isn't initialized yet, return default
        if (!this.intentPatterns) {
            return 'STATEMENT';
        }
        
        for (const [intent, pattern] of Object.entries(this.intentPatterns)) {
            if (pattern.test(normalizedText)) {
                return intent;
            }
        }
        
        return 'STATEMENT';
    }

    _extractKeywords(text) {
        const stopwords = new Set([
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'i', 'you', 'we', 'they'
        ]);

        return text
            .toLowerCase()
            .replace(/[^\w\s]/g, '')
            .split(/\s+/)
            .filter(word => word.length > 2 && !stopwords.has(word));
    }

    findBestTemplate(inputText, threshold = 0.3) {
        if (!inputText || this.templates.length === 0) {
            return { bestMatch: null, confidence: 0 };
        }

        let bestMatch = null;
        let bestSimilarity = 0;
        
        const normalizedInput = inputText.toLowerCase().trim();
        const inputIntent = this._detectIntent(inputText);
        const inputKeywords = this._extractKeywords(inputText);

        for (const template of this.templates) {
            // Calculate base similarity using Levenshtein distance
            const distance = levenshteinDistance(normalizedInput, template.input);
            const maxLen = Math.max(normalizedInput.length, template.input.length);
            let similarity = 1 - (distance / maxLen);

            // Boost similarity if intents match
            if (template.intent === inputIntent) {
                similarity *= 1.3;
            }

            // Boost similarity based on keyword overlap
            const keywordOverlap = inputKeywords.filter(kw => 
                template.keywords.includes(kw)
            ).length;
            
            if (keywordOverlap > 0) {
                const overlapRatio = keywordOverlap / Math.max(inputKeywords.length, 1);
                similarity *= (1 + overlapRatio * 0.5);
            }

            // Check for exact substring matches (high priority)
            if (template.input.includes(normalizedInput) || normalizedInput.includes(template.input)) {
                similarity *= 1.2;
            }

            // Update best match if this is better
            if (similarity >= threshold && similarity > bestSimilarity) {
                bestSimilarity = similarity;
                bestMatch = {
                    input: template.input,
                    output: template.output,
                    intent: template.intent,
                    confidence: Math.min(similarity, 1.0) // Cap at 1.0
                };
            }
        }
        
        return { 
            bestMatch, 
            confidence: bestSimilarity 
        };
    }

    findSimilarTemplates(inputText, limit = 5, threshold = 0.2) {
        if (!inputText || this.templates.length === 0) {
            return [];
        }

        const normalizedInput = inputText.toLowerCase().trim();
        const inputIntent = this._detectIntent(inputText);
        const inputKeywords = this._extractKeywords(inputText);

        const matches = this.templates
            .map(template => {
                const distance = levenshteinDistance(normalizedInput, template.input);
                const maxLen = Math.max(normalizedInput.length, template.input.length);
                let similarity = 1 - (distance / maxLen);

                if (template.intent === inputIntent) {
                    similarity *= 1.2;
                }

                const keywordOverlap = inputKeywords.filter(kw => 
                    template.keywords.includes(kw)
                ).length;
                
                if (keywordOverlap > 0) {
                    const overlapRatio = keywordOverlap / Math.max(inputKeywords.length, 1);
                    similarity *= (1 + overlapRatio * 0.3);
                }

                return {
                    ...template,
                    confidence: Math.min(similarity, 1.0)
                };
            })
            .filter(match => match.confidence >= threshold)
            .sort((a, b) => b.confidence - a.confidence)
            .slice(0, limit);

        return matches;
    }

    addTemplate(input, output) {
        const template = {
            input: input.toLowerCase(),
            output: output,
            intent: this._detectIntent(input),
            keywords: this._extractKeywords(input)
        };

        this.templates.push(template);

        // Save to training data
        if (!this.trainingData.conversations) {
            this.trainingData.conversations = [];
        }

        this.trainingData.conversations.push({
            input: input,
            output: output,
            timestamp: new Date().toISOString()
        });

        this._saveTrainingData();
    }

    _saveTrainingData() {
        try {
            fs.writeFileSync(
                this.trainingDataPath, 
                JSON.stringify(this.trainingData, null, 2)
            );
        } catch (error) {
            console.error('Error saving training data:', error);
        }
    }

    getTemplateStats() {
        const intentCounts = {};
        this.templates.forEach(template => {
            intentCounts[template.intent] = (intentCounts[template.intent] || 0) + 1;
        });

        return {
            totalTemplates: this.templates.length,
            intentDistribution: intentCounts,
            averageKeywords: this.templates.reduce((sum, t) => sum + t.keywords.length, 0) / this.templates.length
        };
    }
}

module.exports = TemplateMatcher;
module.exports.levenshteinDistance = levenshteinDistance;
