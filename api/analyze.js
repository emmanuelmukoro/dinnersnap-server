// api/analyze.js
// ESM-only Node function for Vercel.
// Vision -> cleaned pantry -> Spoonacular + LLM -> emergency fallback.

export const config = {
  runtime: "nodejs",
  maxDuration: 25,
  memory: 1024,
  regions: ["lhr1"],
};

/* ----------------------- Env ----------------------- */
const GCV_KEY = process.env.GCV_KEY;
const SPOON_KEY = process.env.SPOON_KEY || process.env.SPOONACULAR_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

/* ----------------------- Helpers ----------------------- */
const json = (status, obj) =>
  new Response(JSON.stringify(obj), {
    status,
    headers: { "content-type": "application/json" },
  });

const nowMs = () => Number(process.hrtime.bigint() / 1000000n);

function withTimeout(run, ms, label = "timeout") {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  return Promise.race([
    run(controller.signal),
    new Promise((_, rej) =>
      setTimeout(() => rej(new Error(label)), ms + 10)
    ),
  ]).finally(() => clearTimeout(timer));
}

function tfetch(url, opts = {}, ms = 2500, label = "fetch-timeout") {
  return withTimeout(
    (signal) => fetch(url, { ...opts, signal }),
    ms,
    label
  );
}

// Read raw body from Node.js request with timeout + length guard
function readBodyNode(
  req,
  { timeoutMs = 3000, maxLength = 3_500_000 } = {}
) {
  return new Promise((resolve, reject) => {
    let body = "";
    const timer = setTimeout(() => {
      reject(new Error("body-timeout"));
    }, timeoutMs);

    req.on("data", (chunk) => {
      body += chunk;
      if (maxLength && body.length > maxLength) {
        clearTimeout(timer);
        reject(new Error("payload-too-large"));
      }
    });

    req.on("end", () => {
      clearTimeout(timer);
      resolve(body);
    });

    req.on("error", (err) => {
      clearTimeout(timer);
      reject(err);
    });
  });
}

/* ----------------------- Classifiers ----------------------- */
const DESSERT = new Set([
  "dessert",
  "pudding",
  "ice cream",
  "smoothie",
  "shake",
  "cookie",
  "brownie",
  "cupcake",
  "cake",
  "muffin",
  "pancake",
  "waffle",
  "jam",
  "jelly",
  "compote",
  "truffle",
  "fudge",
  "sorbet",
  "parfait",
  "custard",
  "tart",
  "shortbread",
  "licorice",
  "cobbler",
]);
const DRINK = new Set([
  "drink",
  "beverage",
  "mocktail",
  "cocktail",
  "margarita",
  "mojito",
  "spritzer",
  "soda",
  "punch",
  "toddy",
]);
const ALCOHOL = new Set([
  "brandy",
  "rum",
  "vodka",
  "gin",
  "whisky",
  "whiskey",
  "bourbon",
  "tequila",
  "wine",
  "liqueur",
  "amaretto",
  "cognac",
  "port",
  "sherry",
]);
const GENERIC = new Set(["pasta", "rice", "bread", "flour", "sugar", "oil", "salt", "pepper"]);
const MEAT = new Set([
  "chicken",
  "beef",
  "pork",
  "lamb",
  "bacon",
  "ham",
  "turkey",
  "shrimp",
  "prawn",
  "salmon",
  "tuna",
  "fish",
]);
const PASTA = new Set([
  "pasta",
  "spaghetti",
  "macaroni",
  "penne",
  "farfalle",
  "orzo",
  "fusilli",
  "linguine",
  "tagliatelle",
]);

const MAIN_WORDS = new Set(["mushroom", "broccoli", "courgette", "spinach"]);

const titleHasAny = (title, set) => {
  const t = String(title || "").toLowerCase();
  for (const w of set) if (t.includes(w)) return true;
  return false;
};

/* ----------------------- Vision ----------------------- */
async function callVision(imageBase64) {
  if (!GCV_KEY)
    return { ocrTokens: [], labels: [], objects: [], error: "no GCV_KEY" };

  const body = {
    requests: [
      {
        image: {
          content: imageBase64.replace(/^data:image\/\w+;base64,/, ""),
        },
        features: [
          { type: "TEXT_DETECTION", maxResults: 1 },
          { type: "LABEL_DETECTION", maxResults: 10 },
          { type: "OBJECT_LOCALIZATION", maxResults: 10 },
        ],
      },
    ],
  };

  const r = await tfetch(
    `https://vision.googleapis.com/v1/images:annotate?key=${GCV_KEY}`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    },
    8000,
    "vision-timeout"
  );

  const j = await r.json();
  const res = j?.responses?.[0] || {};

  const rawText = (res.textAnnotations?.[0]?.description || "").toLowerCase();
  const RAW_TOKENS = rawText
    .split(/[^a-z]+/g)
    .map((t) => t.trim())
    .filter(Boolean);

  const FOODISH = new Set([
    "ingredients",
    "organic",
    "sauce",
    "soup",
    "canned",
    "tin",
    "dried",
    "fresh",
    "frozen",
    "beans",
    "peas",
    "lentils",
    "broccoli",
    "banana",
    "tomato",
    "onion",
    "garlic",
    "squash",
    "chickpea",
    "chickpeas",
    "feta",
    "rice",
    "pasta",
    "spaghetti",
    "oats",
    "milk",
    "yogurt",
    "cheese",
    "lemon",
    "orange",
    "pepper",
    "spinach",
    "mushroom",
    "potato",
    "avocado",
    "coconut",
    "cream",
    "seasoning",
  ]);

  const ocrTokens = RAW_TOKENS.filter(
    (t) => t.length >= 3 && FOODISH.has(t)
  );
  const labels = (res.labelAnnotations || [])
    .map((x) => x.description?.toLowerCase())
    .filter(Boolean);
  const objects = (res.localizedObjectAnnotations || [])
    .map((x) => x.name?.toLowerCase())
    .filter(Boolean);

  return { ocrTokens, labels, objects };
}

/* ----------------------- Pantry helpers ----------------------- */
const WHITELIST = [
  "banana",
  "broccoli",
  "chickpeas",
  "beans",
  "kidney beans",
  "red kidney beans",
  "tomatoes",
  "onion",
  "garlic",
  "ginger",
  "olive oil",
  "pasta",
  "spaghetti",
  "courgette",
  "feta",
  "orzo",
  "rice",
  "egg",
  "spring onion",
  "lemon",
  "orange",
  "avocado",
  "coconut milk",
  "coconut cream",
  "butternut squash",
  "squash",
  "carrot",
  "pepper",
  "bell pepper",
  "spinach",
  "potato",
  "mushroom",
  "cheese",
  "cheddar",
  "yogurt",
  "milk",
  "almond milk",
  "chicken",
  "chicken breast",
  "beef",
  "pork",
  "fish",
  "salmon",
  "tuna",
  "bread",
  "tortilla",
  "wrap",
  "lentils",
  "cucumber",
  "lettuce",
  "cabbage",
  "kale",
  "apple",
  "pear",
  "oats",
  "flour",
  "sugar",
  "butter",
  "salt",
  "pepper",
  "black pepper",
  "stock cube",
  "curry powder",
  "mixed dried herbs",
  "garam masala",
  "maggi seasoning",
  "jeera",
  "cumin",
  "cloves",
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
  brockley: "broccoli",
  maggi: "maggi seasoning",
  "jeera powder": "jeera",
};

const uniqLower = (arr) => [...new Set(arr.map((x) => String(x).toLowerCase()))];

function lev(a, b) {
  const m = [];
  for (let i = 0; i <= b.length; i++) m[i] = [i];
  for (let j = 0; j <= a.length; j++) m[0][j] = j;
  for (let i = 1; i <= b.length; i++) {
    for (let j = 1; j <= a.length; j++) {
      m[i][j] = Math.min(
        m[i - 1][j] + 1,
        m[i][j - 1] + 1,
        m[i - 1][j - 1] + (b[i - 1] === a[j - 1] ? 0 : 1)
      );
    }
  }
  return m[b.length][a.length];
}

function nearestWhitelistStrict(term) {
  let best = null;
  let bestDist = 2;
  for (const w of WHITELIST) {
    const d = lev(term, w);
    if (d < bestDist) {
      bestDist = d;
      best = w;
    }
  }
  return bestDist <= 1 ? best : null;
}

function addPhraseCombos(tokens) {
  const lower = tokens.map((t) => String(t).toLowerCase());
  const joined = lower.join(" ");
  const extra = [];

  if (joined.includes("coconut") && joined.includes("milk")) {
    extra.push("coconut milk");
  }

  if (
    joined.includes("red") &&
    joined.includes("kidney") &&
    joined.includes("beans")
  ) {
    extra.push("kidney beans", "red kidney beans");
  }

  return [...lower, ...extra];
}

function stripMeatWhenSeasoning(tokens) {
  const lower = tokens.map((t) => String(t).toLowerCase());
  const hasSeasoning = lower.some((t) =>
    /seasoning|stock|bouillon|gravy/.test(t)
  );
  if (!hasSeasoning) return lower;

  const meatWords = new Set([
    "beef",
    "chicken",
    "lamb",
    "pork",
    "ham",
    "turkey",
    "steak",
  ]);

  return lower.filter((t) => !meatWords.has(t));
}

function cleanPantry(raw) {
  const out = new Set();
  for (const t0 of uniqLower(raw)) {
    const t = MAP[t0] || t0;
    if (WHITELIST.includes(t)) {
      out.add(t);
      continue;
    }
    if (t.length >= 5) {
      const near = nearestWhitelistStrict(t);
      if (near) {
        out.add(near);
        continue;
      }
    }
  }
  return [...out];
}

/* ----------------------- Spoonacular ----------------------- */
async function spoonacularRecipes(pantry, prefs) {
  if (!SPOON_KEY) return { results: [], info: { note: "no SPOON_KEY" } };

  const hasMeat = pantry.some((p) => MEAT.has(p));
  const include = pantry.join(",");
  const timeCap = Math.max(10, (prefs?.time ?? 25) + 10);

  const url = new URL("https://api.spoonacular.com/recipes/complexSearch");
  url.searchParams.set("apiKey", SPOON_KEY);
  url.searchParams.set("includeIngredients", include);
  url.searchParams.set("instructionsRequired", "true");
  url.searchParams.set("addRecipeInformation", "true");
  url.searchParams.set("sort", "max-used-ingredients");
  url.searchParams.set("number", "18");
  url.searchParams.set("ignorePantry", "true");
  url.searchParams.set("type", "main course");
  url.searchParams.set("excludeIngredients", Array.from(ALCOHOL).join(","));
  url.searchParams.set("maxReadyTime", String(timeCap));
  if (!hasMeat) url.searchParams.set("diet", "vegetarian");

  let j = {};
  try {
    const r = await tfetch(url.toString(), {}, 2000, "spoon-timeout");
    j = await r.json();
  } catch (e) {
    return { results: [], info: { error: String(e?.message || e) } };
  }

  const raw = Array.isArray(j?.results) ? j.results : [];
  const specific = pantry.filter((p) => !GENERIC.has(p));
  const pantrySet = new Set(pantry);
  const staplesSet = new Set(["water", "salt", "pepper", "oil"]);

  const filtered = raw.filter((it) => {
    const title = String(it.title || "").toLowerCase();
    const dish = (it.dishTypes || []).map((d) => String(d).toLowerCase());
    const ingNames = (it.extendedIngredients || []).map((ing) =>
      String(ing.name || "").toLowerCase()
    );

    // basic filters
    if (titleHasAny(title, DESSERT)) return false;
    if (titleHasAny(title, DRINK)) return false;
    if (titleHasAny(title, ALCOHOL)) return false;
    if (
      dish.some((d) =>
        ["drink", "beverage", "cocktail", "dessert"].includes(d)
      )
    )
      return false;

    const used = it.usedIngredientCount ?? 0;
    const time = it.readyInMinutes ?? 999;
    if (used < 2) return false;
    if (time > timeCap) return false;

    // specific overlap
    const matchSpecificCount = specific.filter((sp) =>
      ingNames.some((n) => n.includes(sp))
    ).length;
    if (specific.length >= 3 && matchSpecificCount < 2) return false;

    // no fish dishes if we don't have fish
    if (
      /shrimp|prawn|salmon|tuna|anchovy|sardine/.test(title) &&
      !pantry.some((p) =>
        ["shrimp", "prawn", "salmon", "tuna", "fish"].includes(p)
      )
    ) {
      return false;
    }

    // main word guard
    const titleWords = title.split(/[^a-z]+/g).filter(Boolean);
    for (const w of MAIN_WORDS) {
      if (titleWords.includes(w)) {
        const hasMain = [...pantrySet].some((p) => p.includes(w));
        if (!hasMain) return false;
      }
    }

    // hit / missing / shopping list
    let hit = 0;
    let missing = 0;
    const missingList = [];

    for (const rawName of ingNames) {
      const simple = rawName.replace(/[^a-z ]/g, "").trim();
      if (!simple) continue;

      const inPantry = [...pantrySet].some((p) => simple.includes(p));
      const isStaple = [...staplesSet].some((p) => simple.includes(p));

      if (inPantry) {
        hit++;
      } else if (isStaple) {
        // ignore
      } else {
        missing++;
        missingList.push(simple);
      }
    }

    // relaxed guard: max 2 missing, majority hit
    if (missing > 2) return false;
    if (hit < 2) return false;
    if (hit + missing > 0 && hit / (hit + missing) < 0.6) return false;

    it._dsMissing = missingList;
    return true;
  });

  const baseList = filtered.length ? filtered : raw;

  const scored = baseList.map((it) => {
    const used = it.usedIngredientCount ?? 0;
    const missed = it.missedIngredientCount ?? 0;
    const time = it.readyInMinutes ?? 30;
    const score =
      0.7 * (used / (used + missed + 1)) +
      0.3 * (1 - Math.min(time, 60) / 60);

    const ingObjs = (it.extendedIngredients || []).map((ing) => {
      const nm = String(ing.name || "").toLowerCase();
      return { name: nm, have: pantry.includes(nm) };
    });

    const missingList = Array.isArray(it._dsMissing) ? it._dsMissing : [];

    return {
      id: String(it.id),
      title: it.title,
      time,
      energy: "hob",
      cost: 2.5,
      score: Math.round(score * 100) / 100,
      ingredients: ingObjs,
      steps: (it.analyzedInstructions?.[0]?.steps || []).map((s, i) => ({
        id: `${it.id}-s${i}`,
        text: s.step,
      })),
      badges: ["web"],
      shoppingList: missingList,
    };
  });

  return {
    results: scored,
    info: {
      spoonacularRawCount: raw.length,
      kept: scored.length,
      timeCap,
      usedFallback: !filtered.length && raw.length > 0,
    },
  };
}

/* ----------------------- LLM (minimal for now) ----------------------- */
async function llmRecipes(pantry, prefs) {
  if (!OPENAI_API_KEY)
    return { recipes: [], info: { reason: "no OPENAI_API_KEY" } };

  const body = {
    model: "gpt-4o-mini",
    temperature: 0.4,
    max_tokens: 700,
    response_format: { type: "json_object" },
    messages: [
      {
        role: "system",
        content:
          "You are DinnerSnap, a world-class savoury dinner chef. You ONLY create realistic DINNER recipes using the given pantry ingredients plus basic staples like water, oil, salt and pepper.",
      },
      {
        role: "user",
        content:
          `Given this pantry: ${pantry.join(
            ", "
          )}, create 1–2 DINNER recipes as a JSON object with this shape:\n` +
          `{\n  "recipes": [\n    {\n      "id": "string",\n      "title": "string",\n      "time": 10–60,\n      "cost": 1–5,\n      "energy": "hob" | "oven" | "air fryer",\n      "ingredients": [{ "name": "string" }],\n      "steps": [{ "id": "string", "text": "string" }]\n    }\n  ]\n}\n\n` +
          "Use ONLY the pantry ingredients plus universal basics. Do NOT invent totally new ingredients.",
      },
    ],
  };

  try {
    const r = await tfetch(
      "https://api.openai.com/v1/chat/completions",
      {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${OPENAI_API_KEY}`,
        },
        body: JSON.stringify(body),
      },
      8000,
      "llm-timeout"
    );

    const j = await r.json();

    if (j?.error) {
      console.log("LLM API error:", j.error);
      return { recipes: [], info: { error: j.error.message || "openai-error" } };
    }

    const raw = j?.choices?.[0]?.message?.content;
    if (!raw) {
      console.log("LLM empty content:", j);
      return { recipes: [], info: { error: "no-content" } };
    }

    let parsed;
    try {
      parsed = typeof raw === "string" ? JSON.parse(raw) : raw;
    } catch (e) {
      console.log("LLM JSON parse error:", e, "raw:", raw);
      return { recipes: [], info: { error: "json-parse-error" } };
    }

    const list = Array.isArray(parsed.recipes) ? parsed.recipes : [];
    const cleaned = list.map((rec, idx) => {
      const id = rec.id || `llm-${Date.now()}-${idx}`;
      return {
        id,
        title: rec.title || "LLM recipe",
        time: rec.time ?? 20,
        cost: rec.cost ?? 2,
        energy: rec.energy || "hob",
        ingredients: (rec.ingredients || []).map((i) => {
          const nm = String(i.name || "").toLowerCase();
          return { name: nm, have: pantry.includes(nm) };
        }),
        steps: (rec.steps || []).map((s, i) => ({
          id: s.id || `${id}-s${i}`,
          text: s.text || String(s),
        })),
        badges: ["llm"],
      };
    });

    return { recipes: cleaned, info: { ok: true } };
  } catch (e) {
    console.log("LLM fetch error:", String(e?.message || e));
    return { recipes: [], info: { error: String(e?.message || e) } };
  }
}

/* ----------------------- Emergency fallback ----------------------- */
function emergencyRecipe(pantry) {
  const pantrySet = new Set(pantry);

  const titleBits = [];
  if (pantry.some((p) => p.includes("chickpea"))) titleBits.push("Chickpea");
  if (
    pantry.includes("coconut cream") ||
    pantry.includes("coconut milk")
  )
    titleBits.push("Coconut");
  if (pantry.some((p) => PASTA.has(p))) titleBits.push("Spaghetti");
  const title =
    (titleBits.length ? titleBits.join(" ") : "Pantry") + " Savoury Skillet";

  const baseIngs = [];

  if (!pantrySet.has("onion")) {
    baseIngs.push({
      name: "onion (or onion powder)",
      have: pantrySet.has("onion"),
    });
  }
  if (!pantrySet.has("garlic")) {
    baseIngs.push({
      name: "garlic (or garlic powder)",
      have: pantrySet.has("garlic"),
    });
  }
  if (!pantrySet.has("ginger")) {
    baseIngs.push({
      name: "ginger (or ground ginger)",
      have: pantrySet.has("ginger"),
    });
  }
  if (!pantrySet.has("mixed dried herbs")) {
    baseIngs.push({
      name: "mixed dried herbs",
      have: pantrySet.has("mixed dried herbs"),
    });
  }
  if (!pantrySet.has("stock cube")) {
    baseIngs.push({
      name: "stock cube",
      have: pantrySet.has("stock cube"),
    });
  }
  if (!pantrySet.has("lemon") && !pantrySet.has("vinegar")) {
    baseIngs.push({
      name: "lemon or vinegar",
      have: pantrySet.has("lemon") || pantrySet.has("vinegar"),
    });
  }
  if (!pantrySet.has("salt") && !pantrySet.has("pepper")) {
    baseIngs.push({
      name: "salt & black pepper",
      have:
        pantrySet.has("salt") ||
        pantrySet.has("black pepper") ||
        pantrySet.has("pepper"),
    });
  }

  const uniqPantry = pantry.map((p) => ({ name: p, have: true }));
  const combined = [...uniqPantry, ...baseIngs];

  const seen = new Set();
  const ingredients = [];
  for (const ing of combined) {
    const nm = ing.name.toLowerCase();
    if (seen.has(nm)) continue;
    seen.add(nm);
    ingredients.push(ing);
  }

  const sample = pantry.slice(0, 4).join(", ") || "your pantry items";

  return {
    id: "local-" + Date.now() + "-" + Math.random().toString(36).slice(2, 8),
    title,
    time: 15,
    cost: 2.0,
    energy: "hob",
    ingredients,
    steps: [
      {
        id: "s1",
        text: "Heat oil; add onion, garlic & ginger. Cook 2–3 min.",
      },
      {
        id: "s2",
        text: `Add your selected pantry ingredients (e.g. ${sample}) and simmer 6–8 min.`,
      },
    ],
    badges: ["local"],
  };
}

/* ----------------------- Handler ----------------------- */
export default async function handler(req) {
  const t0 = nowMs();
  if (req.method !== "POST") return json(405, { error: "POST only" });

  // --- SAFE BODY PARSE (NODE REQUEST) WITH TIMEOUT + SIZE GUARD ---
  let rawBody;
try {
  rawBody = await readBodyNode(req, {
    timeoutMs: 3000,
    maxLength: 3_500_000, // ~2.6 MB binary when base64
  });
} catch (e) {
  const msg = String(e?.message || e);
  console.log("body parse error:", msg);
  const fallback = emergencyRecipe([]);
  return json(200, {
    pantry: [],
    recipes: [fallback],
    debug: { source: "body-parse-fail", error: msg },
  });
}

let body;
try {
  body = rawBody ? JSON.parse(rawBody) : {};
} catch (e) {
  const msg = String(e?.message || e);
  console.log("JSON parse error:", msg);
  const fallback = emergencyRecipe([]);
  return json(200, {
    pantry: [],
    recipes: [fallback],
    debug: { source: "json-parse-fail", error: msg },
  });
}

  const { imageBase64, pantryOverride, prefs = {} } = body || {};
  const watchdogMs = 12000;

  const main = async () => {
    const tStart = nowMs();
    let pantry = [];
    let source = "";
    let pantryFrom = {};

    if (Array.isArray(pantryOverride) && pantryOverride.length) {
      pantry = cleanPantry(pantryOverride);
      source = "pantryOverride";
    } else if (imageBase64) {
      try {
        const v0 = nowMs();
        const visRes = await withTimeout(
          () => callVision(imageBase64),
          2500,
          "vision-timeout"
        );

        pantryFrom = {
          ocr: visRes.ocrTokens,
          labels: visRes.labels,
          objects: visRes.objects,
        };

        const rawTokens = [
          ...(visRes.ocrTokens || []),
          ...(visRes.labels || []),
          ...(visRes.objects || []),
        ];
        const seasoningSafe = stripMeatWhenSeasoning(rawTokens);
        const withCombos = addPhraseCombos(seasoningSafe);
        pantry = cleanPantry(withCombos);
        source = "vision";
        console.log("vision ok in", nowMs() - v0, "ms pantry=", pantry);
      } catch (e) {
        source = "vision-failed";
        pantryFrom = { error: String(e?.message || e) };
        pantry = [];
      }
    } else {
      return json(400, { error: "imageBase64 required (or pantryOverride)" });
    }

    if (prefs?.pantryOnly) {
      const debug = {
        source,
        pantryFrom,
        cleanedPantry: pantry,
        usedLLM: false,
        totalMs: nowMs() - tStart,
        mode: "pantryOnly",
      };
      console.log("analyze pantryOnly debug:", debug);
      return json(200, { pantry, recipes: [], debug });
    }

    const spoonP = (async () => {
      const s0 = nowMs();
      const sp = await spoonacularRecipes(pantry, prefs);
      console.log(
        "spoon end in",
        nowMs() - s0,
        "ms kept=",
        sp.info?.kept,
        "fallback=",
        sp.info?.usedFallback
      );
      return sp.results || [];
    })();

    const llmP = (async () => {
      const l0 = nowMs();
      const llm = await llmRecipes(pantry, prefs);
      console.log(
        "llm end in",
        nowMs() - l0,
        "ms count=",
        (llm.recipes || []).length,
        "info=",
        llm.info
      );
      return llm.recipes || [];
    })();

    let spoonList = [];
    let llmList = [];
    try {
      [spoonList, llmList] = await Promise.all([
        spoonP.catch(() => []),
        llmP.catch(() => []),
      ]);
    } catch {
      // ignore, handled above
    }

    let combined = [];
    if (spoonList.length) combined = spoonList;

    if (llmList.length) {
      for (const rec of llmList) {
        if (!combined.find((r) => r.id === rec.id)) {
          combined.push(rec);
        }
      }
    }

    if (!combined.length) {
      combined = [emergencyRecipe(pantry)];
    }

    if (combined.length > 3) combined = combined.slice(0, 3);

    const debug = {
      source,
      pantryFrom,
      cleanedPantry: pantry,
      usedLLM: combined.some(
        (r) => Array.isArray(r.badges) && r.badges.includes("llm")
      ),
      totalMs: nowMs() - tStart,
    };
    console.log("analyze debug:", debug);

    return json(200, { pantry, recipes: combined, debug });
  };

  try {
    const out = await withTimeout(() => main(), watchdogMs, "watchdog");
    console.log("handler total=", nowMs() - t0, "ms");
    return out;
  } catch (e) {
    console.log("watchdog fired:", String(e?.message || e));
    return json(200, {
      pantry: [],
      recipes: [emergencyRecipe([])],
      debug: { source: "watchdog", error: String(e?.message || e) },
    });
  }
}
