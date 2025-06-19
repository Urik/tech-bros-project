import { query, type SDKMessage } from "@anthropic-ai/claude-code";

async function main() {
  const messages: SDKMessage[] = [];

  console.log("TEST");
  console.log(process.env.ANTHROPIC_API_KEY);
  try {

    for await (const message of query({
      prompt: `
    Analyze this codebase and provide a comprehensive overview.
    Please examine the actual files and provide accurate, specific information. Ensure the output is in valid JSON format.
    `,
      abortController: new AbortController(),
      options: {
        maxTurns: 3,
        cwd: "/Users/uribermankleiner/repos/maintainx",
        executable: "node",
      },
    })) {
      messages.push(message);
    }
  
    console.log(messages);
  } catch (error) {
    console.log(error);
    throw error;
  }
}

main();
