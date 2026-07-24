const readline = require("node:readline");

const lines = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});

function send(id, result) {
  process.stdout.write(`${JSON.stringify({ jsonrpc: "2.0", id, result })}\n`);
}

lines.on("line", (line) => {
  if (!line.trim()) return;
  const request = JSON.parse(line);
  if (request.method === "initialize") {
    send(request.id, {
      protocolVersion: "2025-06-18",
      capabilities: { tools: {} },
      serverInfo: { name: "fixture-node-mcp", version: "1.0.0" },
    });
    return;
  }
  if (request.method === "tools/list") {
    send(request.id, {
      tools: [
        {
          name: "lookup_node_transit_window",
          description: "Query a fictional transit window from a Node MCP.",
          inputSchema: {
            type: "object",
            properties: {
              target: { type: "string" },
            },
          },
        },
      ],
    });
    return;
  }
  if (request.method === "tools/call") {
    send(request.id, {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            status: "success",
            window: "22:05-22:52",
          }),
        },
      ],
    });
  }
});
