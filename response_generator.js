const TemplateMatcher = require('./template_matcher');

class ResponseGenerator {
    constructor(knowledgePath = 'knowledge.json', trainingDataPath = 'training_data.json') {
        this.matcher = new TemplateMatcher(knowledgePath, trainingDataPath);
    }

    generateResponse(inputText) {
        const { bestMatch, score } = this.matcher.findBestTemplate(inputText);
        
        if (!bestMatch) {
            return "I'm not sure how to respond to that.";
        }

        // If we have an exact match or very close match, return the stored response
        if (score < 3) {
            return bestMatch.output;
        }

        // For less exact matches, try to modify the template
        let response = bestMatch.output;
        
        // Replace any placeholders with knowledge base values
        for (const [key, value] of Object.entries(this.matcher.knowledge)) {
            const placeholder = new RegExp(`{${key}}`, 'g');
            response = response.replace(placeholder, value);
        }

        // Check if this was previously repeated
        if (this.lastResponse === response) {
            return `We talked about "${inputText}" earlier. Anything else on your mind?`;
        }

        this.lastResponse = response;
        return response;
    }

    updateKnowledge(key, value) {
        this.matcher.knowledge[key] = value;
    }
}

module.exports = ResponseGenerator;
