// api/analyze.js
// Vision -> cleaned pantry -> Spoonacular (main-course only, no drinks/desserts) -> LLM fallback (savory + seasoned)
export const config = { runtime: "edge" };

const GCV_KEY = process.env.GCV_KEY;
const SPOON_KEY = process.env.SPOON_KEY || process.env.SPOONACULAR_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

const json = (status, obj) =>
  new Response(JSON.stringify(obj), {
    status,
    headers: { "content-type": "application/json" },
  });

// --- classifiers / helpers ---
const DESSERT_WORDS = new Set([
  "dessert","pudding","ice cream","smoothie","shake","cookie","brownie","cupcake",
  "cake","muffin","pancake","waffle","jam","jelly","compote","truffle","fudge",
  "sorbet","parfait","custard","tart","shortbread","licorice","cobbler"
]);

const DRINK_WORDS = new Set([
  "drink","cocktail","beverage","mocktail","soda","punch","spritzer","cobbler","toddy"
]);

const ALCOHOL_WORDS = new Set([
  "brandy","rum","vodka","gin","whisky","whiskey","bourbon","tequila","wine","liqueur","amaretto","cognac","port","sherry"
]);

const MEAT_WORDS = new Set(["chicken","beef","pork","lamb","bacon","ham","turkey","shrimp","prawn","salmon","tuna","fish"]);
const PASTA_WORDS = new Set(["pasta","spaghetti","macaroni","penne","farfalle","orzo","fusilli","linguine","tagliatelle"]);

const GENERIC_WORDS = new Set(["pasta","rice","bread","flour","sugar","oil","salt","pepper"]);

function titleHasAny(title, set) {
  const t = title.toLowerCase();
  for (const w of set) if (t.includes(w)) return true;
  return false;
}

function hasSpecificPantryWordInTitleOrIngredients(item, pantrySpecific) {
  const t = (item.title || "").toLowerCase();
  const ings = (item.extendedIngredients || []).map(i => String(i.name || "").toLowerCase());
  return pantrySpecific.some(p => t.includes(p) || ings.some(n => n.includes(p)));
}

// ---------- Vision ----------
async function callVision(imageBase64) {
  const body = {
    requests: [{
      image: { content: imageBase64.replace(/^data:image\/\w+;base64,/, "") },
      features: [
        { type: "TEXT_DETECTION", maxResults: 1 },
        { type: "LABEL_DETECTION", maxResults: 10 },
        { type: "OBJECT_LOCALIZATION", maxResults: 10 },
      ],
    }],
  };

  const r = await fetch(
    `https://vision.googleapis.com/v1/images:annotate?key=${GCV_KEY}`,
    { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }
  );
  const j = await r.json();
  const res = j?.responses?.[0] || {};

  // OCR tokens -> keep foodish tokens only (less background noise)
  const ocrRaw = (res.textAnnotations?.[0]?.description || "").toLowerCase();
  const RAW_TOKENS = ocrRaw
    .split(/[^a-z]+/g)
    .map(t => t.trim())
    .filter(t => t.length >= 4);

  const FOODISH_HINTS = new Set([
    "ingredients","organic","sauce","soup","canned","tin","dried","fresh","frozen",
    "beans","peas","lentils","broccoli","banana","tomato","onion","garlic","squash",
    "chickpea","chickpeas","feta","rice","pasta","oil","olive","seed","seeds","flour",
    "oats","milk","yogurt","cheese","lemon","pepper","spinach","mushroom","potato",
    "avocado","orange","coconut","cream","spaghetti"
  ]);
  const ocrTokens = RAW_TOKENS.filter(t => FOODISH_HINTS.has(t));

  const labels  = (res.labelAnnotations || []).map(x => x.description?.toLowerCase()).filter(Boolean);
  const objects = (res.localizedObjectAnnotations || []).map(x => x.name?.toLowerCase()).filter(Boolean);

  return { ocrTokens, labels, objects };
}

// ---------- pantry cleanup ----------
const WHITELIST = [
  "banana","broccoli","chickpeas","tomatoes","onion","garlic","olive oil","pasta","spaghetti",
  "courgette","feta","orzo","rice","egg","spring onion","lemon","orange","avocado",
  "butternut squash","squash","carrot","pepper","bell pepper","spinach",
  "potato","mushroom","cheese","yogurt","milk","coconut cream",
  "chicken","chicken breast","beef","pork","fish","salmon","tuna",
  "bread","tortilla","wrap","beans","kidney beans","black beans","lentils",
  "cucumber","lettuce","cabbage","kale","avocado","apple","orange","pear",
  "oats","flour","sugar","butter","salt","pepper"
];

const MAP = {
  bananas: "banana",
  chickpea: "chickpeas",
  garbanzo: "chickpeas",
  "garbanzo bean": "chickpeas",
  "garbanzo beans": "chickpeas",
  zucchini: "courgette",
  courgettes: "courgette",
  tomato: "tomatoes",
  onions: "onion",
  eggs: "egg",
  brockley: "broccoli", // typo
};

function uniqLower(a) { const s = new Set(); for (const x of a) s.add(String(x).toLowerCase()); return [...s]; }
function lev(a, b) {
  const m = [];
  for (let i = 0; i <= b.length; i++) m[i] = [i];
  for (let j = 0; j <= a.length; j++) m[0][j] = j;
  for (let i = 1; i <= b.length; i++) {
    for (let j = 1; j <= a.length; j++) {
      m[i][j] = Math.min(m[i-1][j] + 1, m[i][j-1] + 1, m[i-1][j-1] + (b.charAt(i-1) === a.charAt(j-1) ? 0 : 1));
    }
  }
  return m[b.length][a.length];
}
function nearestWhitelistStrict(term) {
  let best = null, bestDist = 2;
  for (const w of WHITELIST) {
    const d = lev(term, w);
    if (d < bestDist) { bestDist = d; best = w; }
  }
  return bestDist <= 1 ? best : null;
}
function cleanPantry(rawTerms) {
  const out = new Set();
  for (const tRaw of uniqLower(rawTerms)) {
    const t = MAP[tRaw] || tRaw;
    if (WHITELIST.includes(t)) { out.add(t); continue; }
    if (t.length >= 5) {
      const nearest = nearestWhitelistStrict(t);
      if (nearest) { out.add(nearest); continue; }
    }
  }
  return [...out];
}

// ---------- Spoonacular (strict dinner-mode) ----------
async function spoonacularRecipes(pantry, prefs) {
  if (!SPOON_KEY) return { results: [], debug: { note: "no SPOON_KEY" } };

  const hasMeat = pantry.some(p => MEAT_WORDS.has(p));
  const include = pantry.join(",");

  const url = new URL("https://api.spoonacular.com/recipes/complexSearch");
  url.searchParams.set("apiKey", SPOON_KEY);
  url.searchParams.set("includeIngredients", include);
  url.searchParams.set("instructionsRequired", "true");
  url.searchParams.set("addRecipeInformation", "true");
  url.searchParams.set("sort", "max-used-ingredients");
  url.searchParams.set("number", "18");
  url.searchParams.set("ignorePantry", "true");
  url.searchParams.set("type", "main course");              // <- dinner-ish only
  url.searchParams.set("excludeIngredients",               // <- block alcohol
    Array.from(ALCOHOL_WORDS).join(",")
  );
  const timeCap = Math.max(10, (prefs?.time ?? 25) + 10);
  url.searchParams.set("maxReadyTime", String(timeCap));
  if (!hasMeat) url.searchParams.set("diet", "vegetarian");

  const r = await fetch(url.toString());
  const j = await r.json();
  const raw = Array.isArray(j?.results) ? j.results : [];

  // pantry specific (non-generic) list for extra relevance check
  const pantrySpecific = pantry.filter(p => !GENERIC_WORDS.has(p));

  const filtered = raw.filter(it => {
    const title = it.title || "";
    const lowerTitle = title.toLowerCase();
    const dishTypes = (it.dishTypes || []).map(d => String(d).toLowerCase());

    // hard rejects
    if (titleHasAny(lowerTitle, DESSERT_WORDS)) return false;
    if (titleHasAny(lowerTitle, DRINK_WORDS)) return false;
    if (titleHasAny(lowerTitle, ALCOHOL_WORDS)) return false;
    if (dishTypes.some(d => ["drink","beverage","cocktail","dessert"].includes(d))) return false;

    // overlap / time
    const used   = it.usedIngredientCount ?? 0;
    const missed = it.missedIngredientCount ?? 0;
    const time   = it.readyInMinutes ?? 999;
    if (used < 2) return false;                 // require real overlap
    if (time > timeCap) return false;

    // require at least one specific pantry token to appear (title or ingredients)
    if (pantrySpecific.length > 0 && !hasSpecificPantryWordInTitleOrIngredients(it, pantrySpecific)) {
      return false;
    }

    // avoid mac&cheese if no dairy in pantry
    if (lowerTitle.includes("macaroni and cheese")) {
      const hasDairy = pantry.some(p => ["cheese","milk","cream"].includes(p));
      if (!hasDairy) return false;
    }

    // avoid seafood if pantry has no fish
    if ((/shrimp|prawn|salmon|tuna|anchovy|sardine/).test(lowerTitle) &&
        !pantry.some(p => ["shrimp","prawn","salmon","tuna","fish"].includes(p))) {
      return false;
    }

    return true;
  });

  const scored = filtered.map((it) => {
    const used   = it.usedIngredientCount ?? 0;
    const missed = it.missedIngredientCount ?? 0;
    const time   = it.readyInMinutes ?? 30;
    const score  = 0.7 * (used / (used + missed + 1)) + 0.3 * (1 - Math.min(time, 60) / 60);
    return {
      id: String(it.id),
      title: it.title,
      time: time,
      energy: "hob",
      cost: 2.5,
      score: Math.round(score * 100) / 100,
      ingredients: (it.extendedIngredients || []).map((ing) => ({
        name: String(ing.name || "").toLowerCase(),
        have: pantry.includes(String(ing.name || "").toLowerCase()),
      })),
      steps: ((it.analyzedInstructions?.[0]?.steps) || []).map((s, i) => ({
        id: `${it.id}-s${i}`,
        text: s.step,
      })),
      badges: ["web"],
    };
  });

  // if spoonacular still returns oddballs, leave to caller to decide LLM
  return { results: scored, debug: { rawCount: raw.length, kept: scored.length, timeCap } };
}

// ---------- LLM fallback (savory + seasoned) ----------
async function llmRecipe(pantry, prefs) {
  if (!OPENAI_API_KEY) return null;

  const system = `You are “DinnerSnap”—a practical 20-minute dinner helper.
Create a **savory main** (NOT a dessert or drink) using the pantry as the core.
Season boldly with common cupboard items if needed (salt, pepper, garlic, onion, chilli, smoked paprika, cumin, curry powder,
dried herbs, stock cube, soy sauce, lemon). Keep it simple, 5–8 steps, minimal pans.`;

  const user = `Pantry: ${pantry.join(", ")}
Diet: ${prefs?.diet ?? "none"}
Energy: ${prefs?.energyMode ?? "hob"}
Budget: ~£${prefs?.budget ?? 3}/serv
Time: ≤${prefs?.time ?? 25} min

Return JSON with keys: title, time, cost, energy, ingredients[{name,have}], steps[{id,text}], badges[].`;

  const r = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${OPENAI_API_KEY}` },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      temperature: 0.4,
      messages: [{ role: "system", content: system }, { role: "user", content: user }],
    }),
  });
  const j = await r.json();
  const txt = j?.choices?.[0]?.message?.content || "";
  const m = txt.match(/\{[\s\S]*\}$/);
  if (!m) return null;

  try {
    const parsed = JSON.parse(m[0]);
    parsed.id = parsed.id || `llm-${Date.now()}`;
    parsed.energy = parsed.energy || "hob";
    parsed.cost = parsed.cost ?? 2.0;
    parsed.badges = Array.isArray(parsed.badges) ? parsed.badges : ["llm"];
    parsed.ingredients = (parsed.ingredients || []).map((i) => ({
      name: String(i.name || "").toLowerCase(),
      have: pantry.includes(String(i.name || "").toLowerCase()),
    }));
    parsed.steps = (parsed.steps || []).map((s, i) => ({
      id: s.id || `step-${i}`,
      text: s.text || String(s),
    }));
    return parsed;
  } catch { return null; }
}

// ---------- handler ----------
export default async function handler(req) {
  if (req.method !== "POST") return json(405, { error: "POST only" });

  let body;
  try { body = await req.json(); } catch { return json(400, { error: "invalid JSON body" }); }

  const { imageBase64, pantryOverride, prefs = {} } = body || {};
  let pantry = [];
  let source = "";

  if (Array.isArray(pantryOverride) && pantryOverride.length) {
    pantry = cleanPantry(pantryOverride);
    source = "pantryOverride";
  } else if (imageBase64) {
    const res = await callVision(imageBase64);
    const fromVision = [...(res.ocrTokens||[]), ...(res.labels||[]), ...(res.objects||[])];
    pantry = cleanPantry(fromVision);
    source = "vision";
  } else {
    return json(400, { error: "imageBase64 required (or pantryOverride)" });
  }

  // SPOONACULAR (strict)
  const sp = await spoonacularRecipes(pantry, prefs);
  let recipes = sp.results || [];

  // Decide if Spoonacular is acceptable; otherwise prefer LLM
  const topLooksBad =
    recipes.length === 0 ||
    titleHasAny((recipes[0]?.title || "").toLowerCase(), DESSERT_WORDS) ||
    titleHasAny((recipes[0]?.title || "").toLowerCase(), DRINK_WORDS) ||
    titleHasAny((recipes[0]?.title || "").toLowerCase(), ALCOHOL_WORDS) ||
    (recipes[0]?.time ?? 999) > Math.max(10, (prefs?.time ?? 25) + 10);

  let usedLLM = false;
  if (topLooksBad) {
    const r = await llmRecipe(pantry, prefs);
    if (r) { recipes = [r]; usedLLM = true; }
  }

  const debug = {
    source,
    cleanedPantry: pantry,
    spoonacularKept: sp.debug?.kept ?? 0,
    spoonacularRaw: sp.debug?.rawCount ?? 0,
    timeCap: sp.debug?.timeCap,
    usedLLM,
  };

  return json(200, { pantry, recipes, debug });
}
