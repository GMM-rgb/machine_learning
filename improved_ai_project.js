const readline = require('readline');
const fs = require('fs');
const https = require('https');

// File to save AI's knowledge
const knowledgeFile = 'knowledge.json';
const imageTrainingDataFile = 'image_training_data.json';

// Load or initialize AI's knowledge
let knowledge = {};
if (fs.existsSync(knowledgeFile)) {
    knowledge = JSON.parse(fs.readFileSync(knowledgeFile, 'utf8'));
} else {
    fs.writeFileSync(knowledgeFile, JSON.stringify(knowledge, null, 2));
}

// Load or initialize image training data
let imageTrainingData = {};
if (fs.existsSync(imageTrainingDataFile)) {
    imageTrainingData = JSON.parse(fs.readFileSync(imageTrainingDataFile, 'utf8'));
} else {
    fs.writeFileSync(imageTrainingDataFile, JSON.stringify(imageTrainingData, null, 2));
}

// Save AI's knowledge to file
const saveKnowledge = () => {
    fs.writeFileSync(knowledgeFile, JSON.stringify(knowledge, null, 2));
    console.log("Knowledge saved!");
};

// Save image training data to file
const saveImageTrainingData = () => {
    fs.writeFileSync(imageTrainingDataFile, JSON.stringify(imageTrainingData, null, 2));
    console.log("Image training data saved!");
};

// Simulated local training accuracy with learning
const trainOnImages = (rounds = 100) => {
    console.log(`Starting training for ${rounds} rounds on 480p images for text detection...`);

    // Simulated image dataset
    const imageSet = [
        { image: "Image1", hasText: true },
        { image: "Image2", hasText: false },
        { image: "Image3", hasText: true },
        { image: "Image4", hasText: false },
        { image: "Image5", hasText: false },
        { image: "Image6", hasText: true },
        { image: "Image7", hasText: false },
    ];

    // Training loop
    for (let round = 1; round <= rounds; round++) {
        console.log(`\nRound ${round}...`);
        let correctGuesses = 0;

        imageSet.forEach((img) => {
            const previousData = imageTrainingData[img.image] || { successes: 0, failures: 0 };

            // AI logic: Adjust guesses based on past performance
            const guess = Math.random() > 0.5 || previousData.successes > previousData.failures;

            // Track results
            const isCorrect = guess === img.hasText;
            if (isCorrect) {
                previousData.successes++;
                correctGuesses++;
            } else {
                previousData.failures++;
            }

            // Save updated training data
            imageTrainingData[img.image] = previousData;
        });

        // Calculate accuracy
        const accuracy = ((correctGuesses / imageSet.length) * 100).toFixed(2);
        console.log(`Round ${round} complete! Accuracy: ${accuracy}%`);

        // Save progress after each round
        saveImageTrainingData();
    }

    console.log("Training completed for all rounds.");
};

// Function to fetch repositories from GitHub based on a query
const fetchGitHubRepos = (language) => {
    return new Promise((resolve, reject) => {
        const url = `https://api.github.com/search/repositories?q=language:${encodeURIComponent(language)}&sort=stars&order=desc&per_page=5`;
        
        const options = {
            headers: {
                'User-Agent': 'Mozilla/5.0'
            }
        };

        https.get(url, options, (res) => {
            let data = '';
            res.on('data', (chunk) => (data += chunk));
            res.on('end', () => {
                try {
                    const result = JSON.parse(data);
                    resolve(result.items);  // Return the top 5 most starred repositories
                } catch (error) {
                    reject(error);
                }
            });
        }).on('error', (err) => reject(err));
    });
};

// Function to parse the README file and gather useful data from GitHub repositories
const parseGitHubRepo = (repo) => {
    return new Promise((resolve, reject) => {
        https.get(repo.readme_url, (res) => {
            let data = '';
            res.on('data', (chunk) => (data += chunk));
            res.on('end', () => {
                // Process the README file content to extract useful keywords or code snippets
                const extractedData = extractKeywordsAndCode(data);
                resolve(extractedData);
            });
        }).on('error', (err) => reject(err));
    });
};

// Function to extract keywords and code from the README file (simplified for the example)
const extractKeywordsAndCode = (readmeContent) => {
    const keywords = [];
    const codeSnippets = [];

    // A very simple regex to capture some code blocks (adjust for your use case)
    const codeRegex = /```(.*?)(\n|.)*?```/g;
    let match;
    while ((match = codeRegex.exec(readmeContent)) !== null) {
        codeSnippets.push(match[0].trim());
    }

    // Extract common keywords (this is a simplified example)
    const keywordRegex = /\b(class|function|variable|loop|if|else|return|const|let|var|async|await|import|export|module|promise)\b/g;
    let keywordMatch;
    while ((keywordMatch = keywordRegex.exec(readmeContent)) !== null) {
        if (!keywords.includes(keywordMatch[0])) {
            keywords.push(keywordMatch[0]);
        }
    }

    return { keywords, codeSnippets };
};

// Save the collected knowledge to a dictionary file
const saveKnowledgeFromGitHub = (language, data) => {
    const filePath = `knowledge_${language}.json`;
    let knowledge = {};

    if (fs.existsSync(filePath)) {
        knowledge = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    }

    knowledge.keywords = data.keywords;
    knowledge.codeSnippets = data.codeSnippets;

    fs.writeFileSync(filePath, JSON.stringify(knowledge, null, 2));
    console.log(`Knowledge saved for ${language} in ${filePath}`);
};

// Function to generate code for JavaScript or Python
const generateCode = (codeKind, description) => {
    let codeSnippet = "";

    if (codeKind.toLowerCase() === "javascript") {
        // Example of generating random number generation in JavaScript
        if (description.includes('random number generation')) {
            codeSnippet = `
const randomNumber = Math.floor(Math.random() * 100); // Generates a random number between 0 and 99
console.log("Random number:", randomNumber);`;
        } else {
            codeSnippet = `
console.log("Here's a simple JavaScript example: ");
console.log("${description}");`;
        }
    } else if (codeKind.toLowerCase() === "python") {
        // Example of generating random number generation in Python
        if (description.includes('random number generation')) {
            codeSnippet = `
import random
random_number = random.randint(0, 99)  # Generates a random number between 0 and 99
print("Random number:", random_number)`;
        } else {
            codeSnippet = `
print("Here's a simple Python example: ")
print("${description}")`;
        }
    } else {
        codeSnippet = `I don't know how to generate code for "${codeKind}". Please specify another language.`;
    }

    // Save generated code and knowledge to `knowledge.json`
    if (!knowledge[codeKind]) {
        knowledge[codeKind] = {};
    }

    knowledge[codeKind][description] = codeSnippet;
    saveKnowledge();  // Save updated knowledge to file

    return codeSnippet;
};

// Respond to a phrase based on learned knowledge
const respond = (rl, phrase, continueConversation) => {
    const lowerPhrase = phrase.toLowerCase();

    const bestMatch = Object.keys(knowledge).find((key) => key.includes(lowerPhrase));
    if (bestMatch) {
        console.log(`AI says: ${knowledge[bestMatch]}`);
        continueConversation(rl);
    } else {
        console.log("I don't know how to respond to that. Would you like to teach me? (Y/N)");
        rl.question("> ", (answer) => {
            if (answer.toLowerCase() === 'y') {
                rl.question("What should I respond with? ", (response) => {
                    knowledge[phrase.toLowerCase()] = response;
                    saveKnowledge();
                    continueConversation(rl);
                });
            } else {
                console.log("Okay, let's continue the conversation.");
                continueConversation(rl);
            }
        });
    }
};

// Continuous talking mode
const talkToAI = (rl) => {
    rl.question("Say something to the AI (or type 'exit' to quit): ", (phrase) => {
        if (phrase.toLowerCase() === 'exit') {
            console.log("Exiting chat mode...");
            main();  // Go back to main menu
        } else {
            respond(rl, phrase, talkToAI);
        }
    });
};

// Function to clear the console
const clearConsole = () => {
    if (process.stdout.clearScreenDown) {
        process.stdout.clearScreenDown();
    } else {
        console.clear();
    }
};

// Main function
const main = () => {
    clearConsole();  // Clear console before showing options
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });

    rl.question("What do you want to do? (1: Teach AI, 2: Talk to AI, 3: Quit, 4: Train AI Locally, 5: Train AI Online, 6: Generate Code, 7: Train AI with GitHub, 8: Train AI on Images, 9: Teach Code): ", (choice) => {
        if (choice === '1') {
            rl.question("Enter a phrase: ", (phrase) => {
                rl.question("What should I respond with? ", (response) => {
                    knowledge[phrase.toLowerCase()] = response;
                    saveKnowledge();
                    main();  // Return to main menu
                });
            });
        } else if (choice === '2') {
            console.log("Entering conversation mode. Type 'exit' to quit.");
            talkToAI(rl);
        } else if (choice === '3') {
            console.log("Goodbye!");
            rl.close();
        } else if (choice === '4') {
            trainOnImages(5); // Train for 5 rounds
            main();  // Return to main menu
        } else if (choice === '5') {
            rl.question("What topic should I train the AI on? ", async (topic) => {
                await trainAIOnline(topic);
                main();  // Return to main menu
            });
        } else if (choice === '6') {
            rl.question("Which programming language would you like to generate code for? (JavaScript/Python): ", (codeKind) => {
                rl.question("Describe the code you want to generate: ", (description) => {
                    const generatedCode = generateCode(codeKind, description);
                    console.log(`Generated Code: \n${generatedCode}`);
                    main();  // Return to main menu
                });
            });
        } else if (choice === '7') {
            rl.question("What programming language would you like to learn from GitHub? (e.g., JavaScript, Python): ", async (language) => {
                try {
                    const repos = await fetchGitHubRepos(language);
                    if (repos.length > 0) {
                        for (const repo of repos) {
                            console.log(`Processing repo: ${repo.name}`);
                            const repoData = await parseGitHubRepo(repo);
                            saveKnowledgeFromGitHub(language, repoData);
                        }
                    } else {
                        console.log("No repositories found for the specified language.");
                    }
                    main();  // Return to main menu
                } catch (err) {
                    console.error(err);
                    main();  // Return to main menu in case of an error
                }
            });
        } else if (choice === '8') {
            rl.question("Enter the number of rounds for image training: ", (rounds) => {
                trainOnImages(parseInt(rounds, 10) || 100);
                main();  // Return to main menu
            });
        } else if (choice === '9') {
            rl.question("What coding language would you like to teach the AI? (JavaScript/Python): ", (language) => {
                rl.question("Describe the code you want to teach the AI: ", (description) => {
                    rl.question("Provide the code for this description: ", (code) => {
                        if (!knowledge[language]) {
                            knowledge[language] = {};
                        }
                        knowledge[language][description] = code;
                        saveKnowledge();  // Save knowledge after teaching
                        console.log(`AI has learned how to code for "${description}"`);
                        main();  // Return to main menu
                    });
                });
            });
        } else {
            console.log("Invalid choice. Please try again.");
            main();  // Return to main menu
        }
    });
};

// Run the program
main();
