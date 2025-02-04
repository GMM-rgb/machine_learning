function levenshteinDistance(s1, s2) {
    if (s1.length < s2.length) {
        return levenshteinDistance(s2, s1);
    }

    if (s2.length === 0) {
        return s1.length;
    }

    let previousRow = Array.from({ length: s2.length + 1 }, (_, i) => i);

    for (let i = 0; i < s1.length; i++) {
        const currentRow = [i + 1];
        for (let j = 0; j < s2.length; j++) {
            const insertions = previousRow[j + 1] + 1;
            const deletions = currentRow[j] + 1;
            const substitutions = previousRow[j] + (s1[i] !== s2[j] ? 1 : 0);
            currentRow.push(Math.min(insertions, deletions, substitutions));
        }
        previousRow = currentRow;
    }

    return previousRow[previousRow.length - 1];
}

function getLevenshteinDistance(a, b) {
    const tmp = Array(b.length + 1)
      .fill(null)
      .map(() => Array(a.length + 1).fill(0));
  
    for (let i = 0; i <= b.length; i++) {
      tmp[i][0] = i;
    }
    for (let j = 0; j <= a.length; j++) {
      tmp[0][j] = j;
    }
  
    for (let i = 1; i <= b.length; i++) {
      for (let j = 1; j <= a.length; j++) {
        tmp[i][j] =
          b[i - 1] === a[j - 1]
            ? tmp[i - 1][j - 1]
            : Math.min(tmp[i - 1][j - 1] + 1, tmp[i][j - 1] + 1, tmp[i - 1][j] + 1);
      }
    }
    console.log("Levenshtein Distance:", tmp[b.length][a.length]);
    if (a.length > 1000 || b.length > 1000) {
      console.warn("Strings are too long, skipping computation.");
      return -1;
    }
    return tmp[b.length][a.length]; // Final distance
  }
  
  // Find the best match in knowledge (with exact and fuzzy match)
  function findBestMatch(query, knowledge) {
    let closestMatch = null;
    let minDistance = Infinity;
  
    // Check for exact matches first
    for (const key in (knowledge)) {
      const normalizedKey = key.toLowerCase();
      if (query === normalizedKey) {
        return knowledge[key];
      }
    }
  
    // Use fuzzy matching based on Levenshtein distance
    for (const key in (knowledge)) {
      const distance = getLevenshteinDistance(query, key);
      if (distance < minDistance) {
        minDistance = distance;
        closestMatch = key;
      }
    }
  
    return closestMatch
      ? knowledge[closestMatch]
      : "Sorry, I didn't understand that.";
  }

module.exports = { levenshteinDistance };
