function formatReport(stats, asJson) {
  if (asJson) {
    return JSON.stringify(stats);
  }
  return `${stats.lines} lines, ${stats.words} words, ${stats.chars} chars`;
}

module.exports = { formatReport };
