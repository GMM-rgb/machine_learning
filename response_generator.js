const TemplateMatcher = require('./template_matcher');
const tf = require('@tensorflow/tfjs-node');

class ResponseGenerator {
    constructor(knowledgePath = 'knowledge.json', trainingDataPath = 'training_data.json') {
        this.matcher = new TemplateMatcher(knowledgePath, trainingDataPath);
        this.model = null;
        this.vocab = {};
        this.loadModel();
    }

    async loadModel() {
        try {
            this.model = await tf.loadLayersModel('file://D:/machine_learning/model.json');
            console.log("Model loaded successfully in ResponseGenerator");
        } catch (error) {
            console.error("Error loading model in ResponseGenerator:", error);
        }
    }

    async generateResponse(inputText) {
        // Start background learning for unknown words
        this.learnInBackground(inputText);

        // Try template matching first
        const { bestMatch, score } = this.matcher.findBestTemplate(inputText);
        
        // If we have a good template match, use it
        if (bestMatch && score < 3) {
            return bestMatch.output;
        }

        // Otherwise, use the transformer model
        if (this.model) {
            try {
                const response = await this.generateModelResponse(inputText);
                if (response) {
                    return response;
                }
            } catch (error) {
                console.error("Error generating model response:", error);
            }
        }

        // Fallback to template response if model fails
        return bestMatch ? bestMatch.output : "I'm not sure how to respond to that.";
    }

    async generateModelResponse(inputText) {
        const words = inputText.toLowerCase().split(/\s+/);
        const inputSequence = words.map(word => this.vocab[word] || 0);
        
        const inputTensor = tf.tensor2d([inputSequence], [1, inputSequence.length]);
        const prediction = this.model.predict(inputTensor);
        const probabilities = await prediction.array();
        
        // Convert prediction to words
        const responseIndices = probabilities[0]
            .map((prob, index) => ({ prob, index }))
            .sort((a, b) => b.prob - a.prob)
            .slice(0, 5)
            .map(item => item.index);

        // Convert indices back to words using reverse vocab lookup
        const reverseVocab = Object.fromEntries(
            Object.entries(this.vocab).map(([word, index]) => [index, word])
        );

        const responseWords = responseIndices
            .map(index => reverseVocab[index])
            .filter(word => word);

        return responseWords.join(' ');
    }

    async learnInBackground(inputText) {
        const words = inputText.toLowerCase().split(/\s+/);
        const unknownWords = words.filter(word => !this.vocab[word]);

        if (unknownWords.length > 0) {
            console.log("Learning new words in background:", unknownWords);
            
            for (const word of unknownWords) {
                // Add to vocabulary with new index
                this.vocab[word] = Object.keys(this.vocab).length + 1;
                
                try {
                    // Simulate getting definition or information about the word
                    const response = await fetch(`https://api.dictionaryapi.dev/api/v2/entries/en/${word}`);
                    const data = await response.json();
                    
                    if (data && data[0] && data[0].meanings) {
                        // Store the learning result
                        this.matcher.updateKnowledge(word, data[0].meanings[0].definitions[0].definition);
                        console.log(`Learned new word: ${word}`);
                    }
                } catch (error) {
                    console.log(`Failed to learn about word: ${word}`);
                }
            }
        }
    }

    updateKnowledge(key, value) {
        this.matcher.knowledge[key] = value;
    }
}

module.exports = ResponseGenerator;
