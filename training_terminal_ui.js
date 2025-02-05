const ResponseGenerator = require('./response_generator');

async function main() {
    const generator = new ResponseGenerator();
    await generator.startConsoleInterface();
}

main().catch(console.error);
