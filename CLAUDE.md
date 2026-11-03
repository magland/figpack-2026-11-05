# Recipe-driven interactive tutorial

The human writes `recipe.md`; Claude builds the app in `app/` from it. The goal is to show the content was crafted by a human.

## Recipe

- The recipe is the source of truth. On "update the app", diff `recipe.md` and apply exactly those changes.
- Don't edit the recipe unless asked, and then keep the edit minimal.
- Never invent tutorial content (text, explanations, equations). If something is missing, say so.
- Small unspecified details (labels, spacing) are your call. Mention the choice briefly.
- The recipe's notes set app-wide conventions. Apply them to every page.
- Flag mistakes, typos and inconsistencies in the recipe, but leave the rewording to the human.

## Style

Keep replies short and conversational.

## App

- Static `index.html`, `style.css` and `main.js`, with no build step. Bundle libraries locally (pinned versions, under `app/vendor/`) so it works offline.
- `npm run dev` serves the app.
- Use theme-aware CSS variables, and make the layout work on phones.
- Use accurate (symplectic) integrators for physical animations.
- After each change, run `node --check`, confirm the element IDs exist, and say what wasn't checked in a browser.

## Provenance

- The arewehuman extension records keystrokes to `.arewehuman/<file>/<workspace>.awh.jsonl`. Never edit or ignore those files.
- `npm run crafted-link` refreshes the title page's "Crafted by a human" link. Run it before committing.
- Commit only when asked, and include everything.
