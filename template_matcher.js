const fs = require('fs');
const Fuse = require('fuse.js');
const { levenshteinDistance } = require('./utils/string_distance');

class TemplateMatcher {
    constructor(knowledgePath, trainingDataPath) {
        this.knowledge = this._loadJson(knowledgePath);
        this.trainingData = this._loadJson(trainingDataPath);
        this.templates = this._extractTemplates();
    }

    _loadJson(path) {
        return JSON.parse(fs.readFileSync(path, 'utf8'));
    }

    _extractTemplates() {
        const templates = {};
        for (const conversation of this.trainingData) {
            const input = conversation.input.toLowerCase();
            const output = conversation.output;
    //        templates.push({ input, output });
        }
        return templates;
    }

    findBestTemplate(inputText, threshold = 0.7) {
        let bestMatch = null;
        let bestScore = Infinity;
        
        inputText = inputText.toLowerCase();

        for (const template of this.templates) {
            const distance = levenshteinDistance(inputText, template.input);
            const maxLen = Math.max(inputText.length, template.input.length);
            const similarity = 1 - (distance / maxLen);
            
            if (similarity > threshold && distance < bestScore) {
                bestScore = distance;
                bestMatch = template;
            }
        }
        
        return { bestMatch, score: bestScore };
    }
}

module.exports = TemplateMatcher;
