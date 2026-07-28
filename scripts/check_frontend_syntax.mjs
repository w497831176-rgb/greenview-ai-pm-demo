import fs from 'node:fs';

const html = fs.readFileSync(new URL('../frontend/index.html', import.meta.url), 'utf8');
const scripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/gi)]
  .map(match => match[1])
  .filter(source => source.trim());

if (!scripts.length) throw new Error('no inline script found');
for (const source of scripts) new Function(source);
console.log(`PASS: ${scripts.length} frontend script block(s) parsed`);
