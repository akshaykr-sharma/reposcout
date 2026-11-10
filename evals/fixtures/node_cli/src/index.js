const fs = require('fs/promises');

async function countFile(path) {
  const text = await fs.readFile(path, 'utf8');
  const words = text.split(/\s+/).filter(Boolean).length;
  return { lines: text.split('\n').length, words, chars: text.length };
}

module.exports = { countFile };
