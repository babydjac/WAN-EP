import { app } from "../../../scripts/app.js";

// Set a distinctive neon-cyan on deep teal look
const BORDER = "#00E5FF";   // electric cyan
const BACK   = "#0B1F2A";   // deep teal/blue

app.registerExtension({
  name: "wan.expert.prompter.color",
  nodeCreated(node) {
    try {
      // Guard to only affect the WAN 2.2 Expert Prompter
      if (node?.comfyClass === "WAN22ExpertPrompter") {
        node.color = BORDER;
        node.bgcolor = BACK;
      }
    } catch (_) {
      // no-op
    }
  },
});

