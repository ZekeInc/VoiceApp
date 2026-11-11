// items.js — the list of things the child is asked to say.
//
// The list itself lives in voice/data/items.json (edited by hand). This file
// only loads it and checks it against the model's phoneme table, so a typo
// in items.json (say "ae" instead of "æ") is caught at start-up instead of
// silently scoring every attempt as wrong.

const ITEMS_URL = new URL('../data/items.json', import.meta.url);

/**
 * Download items.json.
 * @returns {Promise<{id: string, show: string, say: string, example: string,
 *           target: string[], alsoOk: string[][]}[]>}
 */
export async function loadItems() {
  const res = await fetch(ITEMS_URL);
  if (!res.ok) throw new Error(`Could not download items.json (${res.status})`);
  const { items } = await res.json();
  if (!items?.length) throw new Error('items.json has no items');
  return items;
}

/**
 * Check that every phoneme in every item is one the model knows.
 * Needs the vocab returned by loadModel(), so it runs after the model loads.
 *
 * @throws {Error} naming the first item and phoneme that is not in vocab.json.
 */
export function checkItems(items, vocab) {
  const known = new Set(vocab.tokens);
  for (const item of items) {
    for (const sequence of [item.target, ...item.alsoOk]) {
      for (const token of sequence) {
        if (!known.has(token)) {
          throw new Error(`items.json: "${item.id}" uses "${token}", which the model does not know`);
        }
      }
    }
  }
}
