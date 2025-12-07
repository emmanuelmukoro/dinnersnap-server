// /api/analyze.js
// Node runtime + classic req/res handler so Vercel actually sends a response.

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
const nowMs = () => Number(process.hrtime.bigint() / 1000000n);

function sendJson(res, status, obj) {
  const data = JSON.stringify(obj);
  res.statusCode = status;
  res.setHeader("content-type", "application/json");
  res.setHeader("content-length", Buffer.byteLength(data));
  res.end(data);
}

function withTimeout(run, ms, label = "timeout") {
  return new Promise((resolve, reject) => {
    let finished = false;
    const timer = setTimeout(() => {
      if (finished) return;
      finished = true;
      reject(new Error(label));
    }, ms);

    Promise.resolve()
      .then(run)
      .then((val) => {
        if (finished) return;
        finished = true;
        clearTimeout(timer);
        resolve(val);
      })
      .catch((err) => {
        if (finished) return;
        finished = true;
        clearTimeout(timer);
        reject(err);
      });
  });
}

function tfetch(url, opts = {}, ms = 2500, label = "fetch-timeout") {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);

  return fetch(url, { ...opts, signal: controller.signal })
    .finally(() => clearTimeout(timer))
    .catch((e) => {
      throw new Error(label + ": " + (e?.message || String(e)));
    });
}

// Safe JSON body reader for Node req
async function readJsonBody(req, timeoutMs = 3000) {
  return withTimeout(
    async () => {
      const chunks = [];
      for await (const chunk of req) {
        chunks.push(chunk);
      }
      const bodyStr = Buffer.concat(chunks).toString("utf8");

      if (bodyStr && bodyStr.length > 3_500_000) {
        const err = new Error("payload-too-large");
        err.code = "PAYLOAD_TOO_LARGE";
        throw err;
      }

      if (!bodyStr) return {};
      try {
        return JSON.parse(bodyStr);
      } catch (e) {
        const err = new Error("invalid-json");
        err.code = "INVALID_JSON";
        throw err;
      }
    },
    timeoutMs,
    "body-timeout"
  );
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

/* ----------------------- Vision (2.5s) ----------------------- */
async function callVision(imageBase64) {
  if (!GCV_KEY) {
    return { ocrTokens: [], labels: [], objects: [], error: "no GCV_KEY" };
  }

  const body = {
    requests: [
      {
        image: { content: String(imageBase64 || "").replace(/^data:image\/\w+;base64,/, "") },
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
    2500,
    "vision-timeout"
  );

  const j = await r.json().catch(() => ({}));
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
    "tomatoes",
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
    "egg",
  ]);

  const ocrTokens = RAW_TOKENS.filter((t) => t.length >= 3 && FOODISH.has(t));
  const labels = (res.labelAnnotations || [])
    .map((x) => x.description?.toLowerCase())
    .filter(Boolean);
  const objects = (res.localizedObjectAnnotations || [])
    .map((x) => x.name?.toLowerCase())
    .filter(Boolean);

  return { ocrTokens, labels, objects };
}

/* ----------------------- Pantry cleanup ----------------------- */
const WHITELIST = [
  "banana",
  "broccoli",
  "chickpeas",
  "beans",
  "kidney beans",
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
    j = await r.json().catch(() => ({}));
  } catch (e) {
    console.log("spoon error:", e?.message || e);
    return { results: [], info: { error: String(e?.message || e) } };
  }

  const raw = Array.isArray(j?.results) ? j.results : [];
  const specific = pantry.filter((p) => !GENERIC.has(p));

  const filtered = raw.filter((it) => {
    const title = String(it.title || "").toLowerCase();
    const dish = (it.dishTypes || []).map((d) => String(d).toLowerCase());

    if (Array.from(DESSERT).some((w) => title.includes(w))) return false;
    if (Array.from(DRINK).some((w) => title.includes(w))) return false;
    if (Array.from(ALCOHOL).some((w) => title.includes(w))) return false;
    if (dish.some((d) => ["drink", "beverage", "cocktail", "dessert"].includes(d))) return false;

    const used = it.usedIngredientCount ?? 0;
    const time = it.readyInMinutes ?? 999;
    if (used < 2) return false;
    if (time > timeCap) return false;

    if (specific.length >= 2) {
      const ingNames = (it.extendedIngredients || []).map((ing) =>
        String(ing.name || "").toLowerCase()
      );
      const ok = specific.some((sp) =>
        ingNames.some((n) => n.includes(sp))
      );
      if (!ok) return false;
    }

    if (/shrimp|prawn|salmon|tuna|anchovy|sardine/.test(title)) {
      if (!pantry.some((p) => ["shrimp", "prawn", "salmon", "tuna", "fish"].includes(p))) {
        return false;
      }
    }

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

    return {
      id: String(it.id),
      title: it.title,
      time,
      energy: "hob",
      cost: 2.5,
      score: Math.round(score * 100) / 100,
      ingredients: (it.extendedIngredients || []).map((ing) => {
        const nm = String(ing.name || "").toLowerCase();
        return { name: nm, have: pantry.includes(nm) };
      }),
      steps: (it.analyzedInstructions?.[0]?.steps || []).map((s, i) => ({
        id: `${it.id}-s${i}`,
        text: s.step,
      })),
      badges: ["web"],
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

/* ----------------------- LLM ----------------------- */
async function llmRecipes(pantry, prefs) {
  if (!OPENAI_API_KEY) {
    return { recipes: [], info: { reason: "no OPENAI_API_KEY" } };
  }

  const pantryText = pantry.join(", ");
  const body = {
    model: "gpt-4o-mini",
    temperature: 0.45,
    max_tokens: 700,
    messages: [
      {
        role: "system",
        content:
          "You are DinnerSnap, an expert home cook. You ONLY return valid JSON. Use ONLY the pantry items and at most 2 extra ingredients.",
      },
      {
        role: "user",
        content:
          `Given this pantry: [${pantryText}]\n` +
          "Return ONE savoury DINNER recipe as pure JSON with this exact schema:\n" +
          "{\n" +
          '  "id": "string",\n' +
          '  "title": "string",\n' +
          '  "time": number,\n' +
          '  "cost": number,\n' +
          '  "energy": "hob" | "oven" | "air fryer",\n' +
          '  "ingredients": [ { "name": "string", "have": boolean } ],\n' +
          '  "steps": [ { "id": "string", "text": "string" } ],\n' +
          '  "shoppingList": [ "string" ]\n' +
          "}\n" +
          "No prose. No markdown. JSON only.",
      },
    ],
  };

  let j;
  try {
    const timeoutMs = 11000;
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
      timeoutMs,
      "llm-timeout"
    );
    j = await r.json().catch(() => ({}));
  } catch (e) {
    console.log("LLM fetch error:", e?.message || e);
    return { recipes: [], info: { error: String(e?.message || e) } };
  }

  if (j?.error) {
    console.log("LLM API error:", j.error);
    return { recipes: [], info: { error: j.error.message || "openai-error" } };
  }

  const raw = j?.choices?.[0]?.message?.content;
  if (!raw || typeof raw !== "string") {
    console.log("LLM empty content:", j);
    return { recipes: [], info: { error: "no content" } };
  }

  let parsed;
  try {
    parsed = JSON.parse(raw.trim());
  } catch (e) {
    console.log("LLM JSON parse failed:", e?.message || e, "raw:", raw.slice(0, 120));
    return { recipes: [], info: { error: "json-parse-failed" } };
  }

  const recipes = [];
  const base = parsed && typeof parsed === "object" ? parsed : {};

  const id = String(base.id || `llm-${Date.now()}`);
  const title = String(base.title || "Pantry Dinner");
  const time = Number.isFinite(base.time) ? base.time : 25;
  const cost = Number.isFinite(base.cost) ? base.cost : 2.0;
  const energy = base.energy || "hob";

  const ingredients = Array.isArray(base.ingredients)
    ? base.ingredients.map((i) => {
        const nm = String(i.name || "").toLowerCase();
        return {
          name: nm,
          have: Boolean(
            i.have || pantry.includes(nm)
          ),
        };
      })
    : pantry.map((p) => ({ name: p, have: true }));

  const steps = Array.isArray(base.steps)
    ? base.steps.map((s, i) => ({
        id: s.id || `step-${i}`,
        text: s.text || String(s),
      }))
    : [
        { id: "step-1", text: "Combine your pantry ingredients in a pan." },
        { id: "step-2", text: "Cook until hot and seasoned to taste." },
      ];

  const shoppingList = Array.isArray(base.shoppingList)
    ? base.shoppingList.map((x) => String(x))
    : [];

  recipes.push({
    id,
    title,
    time,
    cost,
    energy,
    ingredients,
    steps,
    badges: ["llm"],
    shoppingList,
  });

  return { recipes, info: { ok: true } };
}

/* ----------------------- Emergency ----------------------- */
function emergencyRecipe(pantry) {
  const titleBits = [];
  if (pantry.some((p) => p.includes("chickpea"))) titleBits.push("Chickpea");
  if (pantry.includes("coconut milk")) titleBits.push("Coconut");
  if (pantry.some((p) => p.includes("pasta") || p.includes("spaghetti"))) titleBits.push("Pasta");
  const title = (titleBits.length ? titleBits.join(" ") : "Pantry") + " Savoury Skillet";

  const baseIngs = [
    { name: "onion (or onion powder)", have: pantry.includes("onion") },
    { name: "garlic (or garlic powder)", have: pantry.includes("garlic") },
    { name: "ginger (or ground ginger)", have: pantry.includes("ginger") },
    { name: "mixed dried herbs", have: pantry.includes("mixed dried herbs") },
    { name: "stock cube", have: pantry.includes("stock cube") },
    {
      name: "lemon or vinegar",
      have: pantry.includes("lemon") || pantry.includes("vinegar"),
    },
    {
      name: "salt & black pepper",
      have: pantry.includes("salt") || pantry.includes("black pepper"),
    },
  ];
  const uniqPantry = pantry.map((p) => ({ name: p, have: true }));
  return {
    id: `local-${Date.now()}`,
    title,
    time: 15,
    cost: 2.0,
    energy: "hob",
    ingredients: [...uniqPantry, ...baseIngs],
    steps: [
      { id: "s1", text: "Heat oil; add onion, garlic & ginger. Cook 2–3 min." },
      { id: "s2", text: "Add pantry items and simmer 6–8 min." },
    ],
    badges: ["local"],
  };
}

/* ----------------------- Core analyze logic ----------------------- */
async function runAnalyze({ imageBase64, pantryOverride, prefs = {}, mode }) {
  const tStart = nowMs();
  let pantry = [];
  let source = "";
  let pantryFrom = {};

  // Pantry from override (manual + essentials)
  if (Array.isArray(pantryOverride) && pantryOverride.length) {
    pantry = cleanPantry(pantryOverride);
    source = "pantryOverride";
  } else if (imageBase64) {
    // Vision path
    try {
      const v0 = nowMs();
      const visRes = await callVision(imageBase64);
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
      pantry = cleanPantry(rawTokens);
      source = "vision";
      console.log("vision ok in", nowMs() - v0, "ms pantry=", pantry);
    } catch (e) {
      source = "vision-failed";
      pantryFrom = { error: String(e?.message || e) };
      pantry = [];
    }
  } else {
    // nothing to analyze
    return {
      pantry: [],
      recipes: [],
      debug: { source: "no-input", totalMs: nowMs() - tStart },
    };
  }

  // Pantry-only fast path
  if (mode === "pantryOnly" || prefs?.pantryOnly) {
    const debug = {
      source,
      pantryFrom,
      cleanedPantry: pantry,
      usedLLM: false,
      totalMs: nowMs() - tStart,
      mode: "pantryOnly",
    };
    console.log("analyze pantryOnly debug:", debug);
    return { pantry, recipes: [], debug };
  }

  // Full providers
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
    const settled = await Promise.allSettled([spoonP, llmP]);
    if (settled[0].status === "fulfilled") spoonList = settled[0].value || [];
    if (settled[1].status === "fulfilled") llmList = settled[1].value || [];
  } catch (e) {
    console.log("provider join error:", e?.message || e);
  }

  let combined = [];
  if (spoonList.length) combined = spoonList;
  if (combined.length < 3 && llmList.length) {
    for (const rec of llmList) {
      if (!combined.find((r) => r.id === rec.id)) {
        combined.push(rec);
        if (combined.length >= 3) break;
      }
    }
  }
  if (!combined.length) combined = [emergencyRecipe(pantry)];
  else if (combined.length > 3) combined = combined.slice(0, 3);

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
  return { pantry, recipes: combined, debug };
}

/* ----------------------- Handler ----------------------- */
export default async function handler(req, res) {
  const t0 = nowMs();

  if (req.method !== "POST") {
    return sendJson(res, 405, { error: "POST only" });
  }

  let body;
  try {
    body = await readJsonBody(req, 3000);
  } catch (e) {
    console.log("body parse error:", e?.message || e);
    if (e.code === "PAYLOAD_TOO_LARGE") {
      return sendJson(res, 413, { error: "payload-too-large" });
    }
    if (e.code === "INVALID_JSON") {
      return sendJson(res, 400, { error: "invalid-json" });
    }
    return sendJson(res, 408, { error: "body-timeout" });
  }

  const { imageBase64, pantryOverride, prefs = {}, mode } = body || {};
  const watchdogMs = 12000;

  try {
    const out = await withTimeout(
      () => runAnalyze({ imageBase64, pantryOverride, prefs, mode }),
      watchdogMs,
      "watchdog"
    );
    console.log("handler total=", nowMs() - t0, "ms");
    return sendJson(res, 200, out);
  } catch (e) {
    console.log("watchdog fired:", String(e?.message || e));
    const fallback = {
      pantry: [],
      recipes: [emergencyRecipe([])],
      debug: { source: "watchdog", error: String(e?.message || e) },
    };
    return sendJson(res, 200, fallback);
  }
}
