const fs = require('fs');
const readline = require('readline');
const jsTokens = require('js-tokens').default;


const pathRoot = '/Users/zerogerc/Documents/datasets/js_dataset.tar/';

const evalInputPath = '/Users/zerogerc/Documents/datasets/js_dataset.tar/programs_training.txt';
const evalOutputPath = '/Users/zerogerc/Documents/datasets/js_dataset.tar/programs_training_tokenized.txt';

function appendLineToFile(filePath, text) {
    fs.appendFileSync(filePath, text + '\n')
}

function readFile(filePath) {
    return fs.readFileSync(filePath, {encoding: 'utf-8'})
}

function processFile(outputPath, filePath) {
    const content = readFile(filePath);
    const tokens = content.match(jsTokens);
    appendLineToFile(outputPath, JSON.stringify(tokens))
}

function main() {
    readline.createInterface({
        input: fs.createReadStream(evalInputPath)
    }).on('line', function (line) {
        if (fs.existsSync(pathRoot + line)) {
            processFile(evalOutputPath, pathRoot + line)
        }
    })
}

main();
