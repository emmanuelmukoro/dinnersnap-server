// api/analyze.js
// Dinnersnap hybrid: Google Vision -> cleaned pantry -> Spoonacular (strict) -> LLM fallback (tasty + seasoned)

export const config = { runtime: "edge" }; // Vercel Edge

const GCV_KEY = process.env.GCV_KEY;
const SPOON_KEY = process.env.SPOON_KEY || process.env.SPOONACULAR_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

// ---------- utilities ----------
const json = (status, obj) =>
  new Response(JSON.stringify(obj), {
    status,
    headers: { "content-type": "application/json" },
  });

const DESSERT_WORDS = new Set([
  "dessert","pudding","ice cream","smoothie","shake","cookie","brownie",
  "cupcake","cake","muffin","pancake","waffle","jam","jelly","compote",
  "truffle","fudge","sorbet","parfait","custard","tart","shortbread",
]);

const MEAT_WORDS = new Set(["chicken","beef","pork","lamb","bacon","ham","turkey","shrimp","prawn","salmon","tuna","fish"]);
const PASTA_WORDS = new Set(["pasta","spaghetti","macaroni","penne","farfalle","orzo","fusilli","linguine","tagliatelle"]);

function titleHasAny(title, set) {
  const t = title.toLowerCase();
  for (const w of set) if (t.includes(w)) return true;
  return false;
}

// ---------- OCR + cleanup ----------
async function callVision(imageBase64) {
  const body = {
    requests: [
      {
        image: { content: imageBase64.replace(/^data:image\/\w+;base64,/, "") },
        features: [
          { type: "TEXT_DETECTION", maxResults: 1 },
          { type: "LABEL_DETECTION", maxResults: 10 },
          { type: "OBJECT_LOCALIZATION", maxResults: 10 },
        ],
      },
    ],
  };

  const r = await fetch(
    `https://vision.googleapis.com/v1/images:annotate?key=${GCV_KEY}`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    }
  );
  const j = await r.json();
  const res = j?.responses?.[0] || {};

  // --- NEW: aggressive OCR token clean-up (keeps "foodish" hints only) ---
  const ocrRaw = (res.textAnnotations?.[0]?.description || "").toLowerCase();
  const RAW_TOKENS = ocrRaw
    .split(/[^a-z]+/g)
    .map((t) => t.trim())
    .filter((t) => t.length >= 4);

  const FOODISH_HINTS = new Set([
    "ingredients","organic","sauce","soup","canned","tin","dried","fresh","frozen",
    "beans","peas","lentils","broccoli","banana","tomato","onion","garlic","squash",
    "chickpea","chickpeas","feta","rice","pasta","oil","olive","seed","seeds","flour",
    "oats","milk","yogurt","cheese","lemon","pepper","spinach","mushroom","potato",
  ]);

  const ocrTokens = RAW_TOKENS.filter((t) => FOODISH_HINTS.has(t));

  const labels = (res.labelAnnotations || []).map((x) => x.description?.toLowerCase()).filter(Boolean);
  const objects = (res.localizedObjectAnnotations || []).map((x) => x.name?.toLowerCase()).filter(Boolean);

  return { ocrTokens, labels, objects };
}

// whitelist + strict fuzzy
const WHITELIST = [
  "banana","broccoli","chickpeas","tomatoes","onion","garlic","olive oil","pasta",
  "courgette","feta","orzo","rice","egg","spring onion","lemon",
  "butternut squash","squash","carrot","pepper","bell pepper","spinach",
  "potato","mushroom","cheese","yogurt","milk",
  "chicken","chicken breast","beef","pork","fish","salmon","tuna",
  "bread","tortilla","wrap","beans","kidney beans","black beans","lentils",
  "cucumber","lettuce","cabbage","kale","avocado","apple","orange","pear",
  "oats","flour","sugar","butter","salt","pepper", "coconut cream"
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
  brockley: "broccoli", // common misspelling you saw
};

function uniqLower(a) {
  const s = new Set();
  for (const x of a) s.add(String(x).toLowerCase());
  return [...s];
}
function lev(a, b) {
  const m = [];
  for (let i = 0; i <= b.length; i++) m[i] = [i];
  for (let j = 0; j <= a.length; j++) m[0][j] = j;
  for (let i = 1; i <= b.length; i++) {
    for (let j = 1; j <= a.length; j++) {
      m[i][j] = Math.min(
        m[i - 1][j] + 1,
        m[i][j - 1] + 1,
        m[i - 1][j - 1] + (b.charAt(i - 1) === a.charAt(j - 1) ? 0 : 1)
      );
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

// ---------- Spoonacular (strict) ----------
async function spoonacularRecipes(pantry, prefs) {
  if (!SPOON_KEY) return { results: [], debug: { note: "no SPOON_KEY" } };

  // Basic diet inference: if no meat words, prefer vegetarian results.
  const hasMeatInPantry = pantry.some((p) => MEAT_WORDS.has(p));
  const include = pantry.join(",");
  const url = new URL("https://api.spoonacular.com/recipes/complexSearch");
  url.searchParams.set("apiKey", SPOON_KEY);
  url.searchParams.set("includeIngredients", include);
  url.searchParams.set("instructionsRequired", "true");
  url.searchParams.set("addRecipeInformation", "true");
  url.searchParams.set("sort", "max-used-ingredients");
  url.searchParams.set("number", "12");
  url.searchParams.set("ignorePantry", "true");
  if (!hasMeatInPantry) url.searchParams.set("diet", "vegetarian");

  const r = await fetch(url.toString());
  const j = await r.json();

  const raw = Array.isArray(j?.results) ? j.results : [];

  // Filter out dessert-ish and require real overlap.
  const filtered = raw.filter((it) => {
    const title = it.title || "";
    if (titleHasAny(title, DESSERT_WORDS)) return false;

    const usedCount = it.usedIngredientCount ?? 0;
    const missedCount = it.missedIngredientCount ?? 0;
    const overlapOK = usedCount >= 2 || usedCount >= Math.min(2, pantry.length); // need some overlap

    // Don't suggest shrimp/salmon/etc. unless pantry has fish
    if (titleHasAny(title, new Set(["shrimp","prawn","salmon","tuna","anchovy","sardine"])) &&
        !pantry.some((p) => ["shrimp","prawn","salmon","tuna","fish"].includes(p))) {
      return false;
    }

    // Avoid “mac and cheese” if we only know generic pasta and no cheese/milk
    if (title.toLowerCase().includes("macaroni and cheese")) {
      const hasDairy = pantry.some((p) => ["cheese","milk","cream"].includes(p));
      if (!hasDairy) return false;
    }

    return overlapOK && missedCount <= 6; // keep realistic
  });

  // Attach a light score for client sorting if you want
  const scored = filtered.map((it) => {
    const used = it.usedIngredientCount ?? 0;
    const missed = it.missedIngredientCount ?? 0;
    const time = it.readyInMinutes ?? 30;
    const score = 0.7 * (used / (used + missed + 1)) + 0.3 * (1 - Math.min(time, 60) / 60);
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

  return { results: scored, debug: { spoonacularRawCount: raw.length, kept: scored.length } };
}

// ---------- LLM fallback ----------
async function llmRecipe(pantry, prefs) {
  if (!OPENAI_API_KEY) return null;

  const system = `You are “DinnerSnap”—a practical 20-minute home-cook assistant.
Write a **savory** recipe (NOT dessert) that uses the user's pantry as the core.
If you need flavour, YOU MAY add common cupboard seasonings even if not listed: 
salt, pepper, garlic, onion, chilli, smoked paprika, cumin, curry powder, dried herbs (oregano/basil/thyme), soy sauce, stock cube, lemon/lime.
Prefer a sensible dinner (stir-fry, curry, pasta, soup, traybake) over sweets.
Keep it simple, 5–8 steps, one pan if you can.`;

  const user = {
    pantry,
    prefs,
    request: "Give me one recipe under 20–25 minutes.",
  };

  const r = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: system },
        {
          role: "user",
          content:
            `Pantry: ${pantry.join(", ")}\nDiet: ${prefs?.diet ?? "none"}\n` +
            `Energy: ${prefs?.energyMode ?? "hob"}\nBudget: ~£${prefs?.budget ?? 3}/serv\nTime: ≤${prefs?.time ?? 25} min\n\n` +
            `Return JSON with keys: title, time, cost, energy, ingredients[{name,have}], steps[{id,text}], badges[].`,
        },
      ],
      temperature: 0.4,
    }),
  });
  const j = await r.json();

  const txt = j?.choices?.[0]?.message?.content || "";
  // extremely lenient JSON capture
  const m = txt.match(/\{[\s\S]*\}$/);
  if (!m) return null;
  try {
    const parsed = JSON.parse(m[0]);
    // ensure id/shape
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
  } catch {
    return null;
  }
}

// ---------- handler ----------
export default async function handler(req) {
  if (req.method !== "POST") return json(405, { error: "POST only" });

  let body;
  try {
    body = await req.json();
  } catch {
    return json(400, { error: "invalid JSON body" });
  }

  const { imageBase64, pantryOverride, prefs = {} } = body || {};
  let pantry = [];
  let source = "";

  if (Array.isArray(pantryOverride) && pantryOverride.length) {
    // caller sends final list directly (from “Find recipes”)
    pantry = cleanPantry(pantryOverride);
    source = "pantryOverride";
  } else if (imageBase64) {
    const res = await callVision(imageBase64);
    const fromVision = [
      ...(res.ocrTokens || []),
      ...(res.labels || []),
      ...(res.objects || []),
    ];
    pantry = cleanPantry(fromVision);
    source = "vision";
  } else {
    return json(400, { error: "imageBase64 required (or pantryOverride)" });
  }

  // Prefer Spoonacular, but **only keep dinner-ish, overlapping matches**.
  const sp = await spoonacularRecipes(pantry, prefs);
  let recipes = sp.results || [];

  // If Spoonacular is empty OR clearly unhelpful (e.g. only 1 match and it’s weird), use LLM.
  const spoonacularAcceptable = recipes.length >= 2 || (recipes.length >= 1 && recipes[0].score >= 0.35);
  let usedLLM = false;
  if (!spoonacularAcceptable) {
    const r = await llmRecipe(pantry, prefs);
    if (r) {
      recipes = [r];
      usedLLM = true;
    }
  }

  // If still nothing, we’ll send pantry only (client can pop a message)
  const debug = {
    source,
    pantryFrom: source === "vision" ? undefined : [],
    cleanedPantry: pantry,
    spoonacularCount: sp.results?.length ?? 0,
    usedLLM,
    extra: sp.debug,
  };

  return json(200, { pantry, recipes, debug });
}
