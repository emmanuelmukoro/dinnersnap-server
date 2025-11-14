// api/analyze.js
// Node.js Vercel Function (ESM).
// Vision -> cleaned pantry -> Spoonacular + LLM (in parallel) -> emergency fallback.

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
const sendJson = (res, status, obj) => {
  res.statusCode = status;
  res.setHeader("content-type", "application/json");
  res.end(JSON.stringify(obj));
};

const nowMs = () => Number(process.hrtime.bigint() / 1000000n);

function withTimeout(run, ms, label = "timeout") {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  return Promise.race([
    run(controller.signal),
    new Promise((_, rej) => setTimeout(() => rej(new Error(label)), ms + 10)),
  ]).finally(() => clearTimeout(timer));
}

function tfetch(url, opts = {}, ms = 2500, label = "fetch-timeout") {
  return withTimeout((signal) => fetch(url, { ...opts, signal }), ms, label);
}

// Read raw body from Node req and JSON.parse it, with a size guard.
function readJsonBody(req, maxChars = 3_500_000) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk;
      if (body.length > maxChars) {
        reject(new Error("payload-too-large"));
        req.destroy();
      }
    });
    req.on("end", () => {
      if (!body) return resolve({});
      try {
        resolve(JSON.parse(body));
      } catch (e) {
        reject(e);
      }
    });
    req.on("error", reject);
  });
}

function stripMeatWhenSeasoning(tokens) {
  const lower = tokens.map((t) => String(t).toLowerCase());
  const hasSeasoningWord = lower.some((t) =>
    /seasoning|stock|bouillon|gravy/.test(t)
  );
  if (!hasSeasoningWord) return tokens;

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


/* ----------------------- Classifiers & constants ----------------------- */
const DESSERT = new Set([
  "dessert","pudding","ice cream","smoothie","shake","cookie","brownie","cupcake","cake",
  "muffin","pancake","waffle","jam","jelly","compote","truffle","fudge","sorbet","parfait",
  "custard","tart","shortbread","licorice","cobbler",
]);
const DRINK   = new Set([
  "drink","beverage","mocktail","cocktail","margarita","mojito","spritzer","soda","punch","toddy",
]);
const ALCOHOL = new Set([
  "brandy","rum","vodka","gin","whisky","whiskey","bourbon","tequila","wine","liqueur",
  "amaretto","cognac","port","sherry",
]);
const GENERIC = new Set(["pasta","rice","bread","flour","sugar","oil","salt","pepper"]);
const MEAT    = new Set(["chicken","beef","pork","lamb","bacon","ham","turkey","shrimp","prawn","salmon","tuna","fish"]);
const PASTA   = new Set(["pasta","spaghetti","macaroni","penne","farfalle","orzo","fusilli","linguine","tagliatelle"]);

// “Main words” that, if in the title but not in pantry, we reject the recipe
const MAIN_WORDS = new Set([
  "mushroom","mushrooms",
  "chicken","beef","pork","lamb","bacon","ham","turkey","sausage",
  "salmon","tuna","prawn","shrimp",
]);

// Staples that we treat as “everyone has” (don’t count as missing)
const ALLOWED_EXTRAS = [
  "water",
  "salt",
  "pepper",
  "oil",
  "olive oil",
  "butter",
  "sugar",
  "flour",
  "stock",
  "stock cube",
];

const titleHasAny = (title, set) => {
  const t = String(title || "").toLowerCase();
  for (const w of set) if (t.includes(w)) return true;
  return false;
};

const hasSpecificTermInTitleOrIngredients = (item, specificTerms) => {
  const t = String(item.title || "").toLowerCase();
  const ings = (item.extendedIngredients || []).map((i) => String(i.name || "").toLowerCase());
  return specificTerms.some((p) => t.includes(p) || ings.some((n) => n.includes(p)));
};

/* ----------------------- Vision (2.5s) ----------------------- */
async function callVision(imageBase64) {
  if (!GCV_KEY)
    return { ocrTokens: [], labels: [], objects: [], error: "no GCV_KEY" };

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

  const j = await r.json();
  const res = j?.responses?.[0] || {};

  const rawText = (res.textAnnotations?.[0]?.description || "").toLowerCase();
  const RAW_TOKENS = rawText
    .split(/[^a-z]+/g)
    .map((t) => t.trim())
    .filter(Boolean);

  const FOODISH = new Set([
    "ingredients","organic","sauce","soup","canned","tin","dried","fresh","frozen","beans","peas","lentils",
    "broccoli","banana","tomato","onion","garlic","squash","chickpea","chickpeas","feta","rice","pasta",
    "spaghetti","oats","milk","yogurt","cheese","lemon","orange","pepper","spinach","mushroom","potato",
    "avocado","coconut","cream",
  ]);

  const ocrTokens = RAW_TOKENS.filter((t) => t.length >= 4 && FOODISH.has(t));
  const labels  = (res.labelAnnotations || []).map((x) => x.description?.toLowerCase()).filter(Boolean);
  const objects = (res.localizedObjectAnnotations || []).map((x) => x.name?.toLowerCase()).filter(Boolean);
  return { ocrTokens, labels, objects };
}

/* ----------------------- Pantry cleanup ----------------------- */
const WHITELIST = [
  "banana","broccoli","chickpeas","beans","kidney beans","tomatoes","onion","garlic","ginger","olive oil",
  "pasta","spaghetti","courgette","feta","orzo","rice","egg","spring onion","lemon","orange","avocado",
  "coconut milk","coconut cream","butternut squash","squash","carrot","pepper","bell pepper","spinach",
  "potato","mushroom","cheese","cheddar","yogurt","milk","almond milk","chicken","chicken breast","beef",
  "pork","fish","salmon","tuna","bread","tortilla","wrap","lentils","cucumber","lettuce","cabbage","kale",
  "apple","pear","oats","flour","sugar","butter","salt","pepper","stock cube","curry powder","mixed dried herbs",
  "garam masala","maggi seasoning","jeera","cumin","cloves",
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

// Combine common multi-word ingredients from OCR tokens
function addPhraseCombos(tokens) {
  const base = tokens
    .map((t) => String(t).toLowerCase())
    .filter(Boolean);
  const set = new Set(base);
  const out = new Set(base);

  // coconut + milk => coconut milk, drop plain milk
  if (set.has("coconut") && set.has("milk")) {
    out.add("coconut milk");
    out.delete("milk");
  }

  // kidney + beans => kidney beans, drop generic beans
  if (set.has("kidney") && set.has("beans")) {
    out.add("kidney beans");
    out.delete("beans");
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
  const staplesSet = new Set(ALLOWED_EXTRAS);

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
    if (dish.some((d) => ["drink", "beverage", "cocktail", "dessert"].includes(d))) return false;

    const used = it.usedIngredientCount ?? 0;
    const time = it.readyInMinutes ?? 999;
    if (used < 2) return false;
    if (time > timeCap) return false;

    // stronger specific overlap
    const matchSpecificCount = specific.filter((sp) =>
      ingNames.some((n) => n.includes(sp))
    ).length;
    if (specific.length >= 3 && matchSpecificCount < 2) return false;

    // no fish dishes if we don't have fish
    if (/shrimp|prawn|salmon|tuna|anchovy|sardine/.test(title) &&
        !pantry.some((p) => ["shrimp","prawn","salmon","tuna","fish"].includes(p))) {
      return false;
    }

    // main words guard: e.g. mushroom recipes require mushrooms
    const titleWords = title.split(/[^a-z]+/g).filter(Boolean);
    for (const w of MAIN_WORDS) {
      if (titleWords.includes(w)) {
        const hasMain = [...pantrySet].some((p) => p.includes(w));
        if (!hasMain) return false;
      }
    }

    // strict-ish pantry vs missing logic
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

    // relaxed mode:
    // - at most 2 missing ingredients
    // - must match majority of non-staple ingredients
    if (missing > 2) return false;
    if (hit < 2) return false;
    if (hit + missing > 0 && hit / (hit + missing) < 0.6) return false;

    // stash missing list for later mapping
    it._dsMissing = missingList;
    return true;
  });

  const scored = filtered.map((it) => {
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
    info: { spoonacularRawCount: raw.length, kept: scored.length, timeCap },
  };
}

/* ----------------------- LLM (JSON mode, fills remaining recipes) ----------------------- */
async function llmRecipes(pantry, prefs, needed = 3) {
  if (!OPENAI_API_KEY || needed <= 0) {
    return { recipes: [], info: { reason: "no OPENAI_API_KEY or needed<=0" } };
  }

  const staples = ALLOWED_EXTRAS;
  const maxRecipes = Math.min(needed, 3);

  const system = [
    "You are DinnerSnap, a world-class chef and recipe developer.",
    "You create simple, tasty DINNER recipes using the ingredients the user already has.",
    "Prefer one-pot or low-faff meals real families can cook on weeknights.",
  ].join(" ");

  const user = [
    "Pantry ingredients (user has these):",
    JSON.stringify(pantry),
    "",
    "Rules:",
    "- You may ONLY use ingredients from that list, plus water, salt, pepper and oil.",
    "- Do not use any other ingredients under any circumstances.",
    "- If the pantry is limited, still produce realistic, comforting recipes.",
    "",
    `Return up to ${maxRecipes} recipes in this JSON format:`,
    '{ "recipes": [ { "id": "string", "title": "string", "time": 20, "cost": 2.5, "energy": "hob", "ingredients": [ { "name": "string" } ], "shoppingList": [], "steps": [ { "id": "s1", "text": "..." } ] } ] }',
  ].join("\n");

  const body = {
    model: "gpt-4o-mini",
    response_format: { type: "json_object" },
    temperature: 0.4,
    max_tokens: 900,
    messages: [
      { role: "system", content: system },
      { role: "user", content: user },
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
      2500,
      "llm-timeout"
    );
    const j = await r.json();
    const raw = j?.choices?.[0]?.message?.content;

if (j?.error) {
  console.log("LLM API error:", j.error);
  return { recipes: [], info: { error: j.error.message || "openai-error" } };
}
if (!raw) {
  console.log("LLM empty content:", j);
  return { recipes: [], info: { error: "no content" } };
}



    let root;
    try {
      root = JSON.parse(raw);
    } catch (e) {
      return { recipes: [], info: { error: "invalid JSON from LLM" } };
    }

    let list = [];
    if (Array.isArray(root?.recipes)) {
      list = root.recipes;
    } else if (root && typeof root === "object") {
      list = [root];
    }

    const out = list.slice(0, maxRecipes).map((rec, idx) => {
      const id = String(rec.id || `llm-${Date.now()}-${idx}`);
      const time = Number.isFinite(rec.time) ? rec.time : 20;
      const cost = Number.isFinite(rec.cost) ? rec.cost : 2.0;
      const energy = typeof rec.energy === "string" ? rec.energy : "hob";

      const ingredients = Array.isArray(rec.ingredients)
        ? rec.ingredients.map((i) => {
            const nm = String(i.name || "").toLowerCase();
            return { name: nm, have: pantry.includes(nm) };
          })
        : pantry.map((p) => ({ name: p, have: true }));

      const steps = Array.isArray(rec.steps)
        ? rec.steps.map((s, i) => ({
            id: s.id || `step-${i}`,
            text: s.text || String(s),
          }))
        : [
            { id: "step-0", text: "Combine your pantry ingredients and cook until done." },
          ];

      const shoppingList = Array.isArray(rec.shoppingList)
        ? rec.shoppingList.map((x) => String(x))
        : [];

      return {
        id,
        title: String(rec.title || "Pantry Dinner"),
        time,
        cost,
        energy,
        ingredients,
        steps,
        badges: ["llm"],
        shoppingList,
      };
    });

    return { recipes: out, info: { ok: true } };
  } catch (e) {
    return { recipes: [], info: { error: String(e?.message || e) } };
  }
}

/* ----------------------- Emergency ----------------------- */
function emergencyRecipe(pantry) {
  const pantrySet = new Set(pantry);

  const titleBits = [];
  if (pantry.some((p) => p.includes("chickpea"))) titleBits.push("Chickpea");
  if (pantry.includes("coconut cream") || pantry.includes("coconut milk"))
    titleBits.push("Coconut");
  if (pantry.some((p) => PASTA.has(p))) titleBits.push("Spaghetti");
  const title =
    (titleBits.length ? titleBits.join(" ") : "Pantry") + " Savoury Skillet";

  const baseIngs = [
    { name: "onion (or onion powder)", have: pantrySet.has("onion") },
    { name: "garlic (or garlic powder)", have: pantrySet.has("garlic") },
    { name: "ginger (or ground ginger)", have: pantrySet.has("ginger") },
    {
      name: "mixed dried herbs",
      have: pantrySet.has("mixed dried herbs"),
    },
    { name: "stock cube", have: pantrySet.has("stock cube") },
    {
      name: "lemon or vinegar",
      have: pantrySet.has("lemon") || pantrySet.has("vinegar"),
    },
    {
      name: "salt & black pepper",
      have:
        pantrySet.has("salt") ||
        pantrySet.has("black pepper") ||
        pantrySet.has("pepper"),
    },
  ];

  // pantry items as {name,have:true}
  const uniqPantry = pantry.map((p) => ({ name: p, have: true }));

  // dedupe by name so we don’t get “salt” twice etc.
  const combined = [...uniqPantry, ...baseIngs];
  const seen = new Set();
  const ingredients = [];
  for (const ing of combined) {
    const nm = ing.name.toLowerCase();
    if (seen.has(nm)) continue;
    seen.add(nm);
    ingredients.push(ing);
  }

  // show a few example pantry items in step text
  const sample = pantry.slice(0, 4).join(", ") || "your pantry items";

  return {
    id:
      "local-" +
      Date.now() +
      "-" +
      Math.random().toString(36).slice(2, 8),
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
    shoppingList: [],
  };
}

/* ----------------------- Handler ----------------------- */
export default async function handler(req, res) {
  const t0 = nowMs();
  if (req.method !== "POST") {
    return sendJson(res, 405, { error: "POST only" });
  }

  let body;
  try {
    body = await readJsonBody(req);
  } catch (e) {
    if (String(e.message) === "payload-too-large") {
      console.log("body too large");
      return sendJson(res, 413, { error: "payload-too-large" });
    }
    console.log("body parse error:", String(e?.message || e));
    return sendJson(res, 400, { error: "invalid JSON body" });
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
      return sendJson(res, 400, {
        error: "imageBase64 required (or pantryOverride)",
      });
    }

    // Pantry-only fast return
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
      return sendJson(res, 200, { pantry, recipes: [], debug });
    }

    // Full providers
    const spoonP = (async () => {
      const s0 = nowMs();
      const sp = await spoonacularRecipes(pantry, prefs);
      console.log("spoon end in", nowMs() - s0, "ms kept=", sp.info?.kept);
      return sp.results || [];
    })();
    const llmP = (async () => {
      const l0 = nowMs();
      const llm = await llmRecipes(pantry, prefs, 3);
      console.log("llm end in", nowMs() - l0, "ms ok=", (llm.recipes || []).length > 0);
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
      // ignore
    }

    let combined = [];

    if (spoonList.length) {
      combined = spoonList.slice(0, 3);
    }

    if (combined.length < 3 && llmList.length) {
      for (const rec of llmList) {
        if (!combined.find((r) => r.id === rec.id)) {
          combined.push(rec);
          if (combined.length >= 3) break;
        }
      }
    }

    // Always ensure 3 recipes
    while (combined.length < 3) {
      combined.push(emergencyRecipe(pantry));
    }

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
    return sendJson(res, 200, { pantry, recipes: combined, debug });
  };

  try {
    await withTimeout(() => main(), watchdogMs, "watchdog");
    console.log("handler total=", nowMs() - t0, "ms");
  } catch (e) {
    console.log("watchdog fired:", String(e?.message || e));
    return sendJson(res, 200, {
      pantry: [],
      recipes: [emergencyRecipe([])],
      debug: { source: "watchdog", error: String(e?.message || e) },
    });
  }
}
