#!/usr/bin/env node
const { program } = require('commander');
const { countFile } = require('./index');
const { formatReport } = require('./report');

program
  .argument('<file>', 'file to count')
  .option('--json', 'print JSON instead of text')
  .action(async (file, options) => {
    const stats = await countFile(file);
    console.log(formatReport(stats, options.json));
  });

program.parse();
