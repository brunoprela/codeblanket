export const propertyBasedTestingQuiz = [
  {
    id: 'pbt-q-1',
    question:
      'Design property-based tests for URL parser: parse_url("https://example.com:443/path?q=1") → {scheme, host, port, path, query}. Define 3-5 properties that must hold for all valid URLs, explain strategies, and show how property-based testing finds bugs example-based tests miss.',
    sampleAnswer:
      'URL parser properties: (1) Round-trip: build_url(parse_url(url)) == url for all valid URLs. Strategy: @given(st.from_regex(r"https?://[^/]+/.*")). Catches: Lost port, query params. (2) Scheme extraction: parse_url(url)["scheme"] in ["http", "https"] for all URLs. Strategy: st.one_of(st.just("http"), st.just("https")). Catches: Invalid schemes. (3) Port defaults: parse_url("http://host")["port"] == 80, parse_url("https://host")["port"] == 443. Strategy: st.sampled_from(["http://host", "https://host"]). Catches: Missing default ports. (4) Path always starts with /: parse_url(url)["path"].startswith("/"). Strategy: st.text(alphabet=st.characters(blacklist_characters="/")). Catches: Missing slash. (5) Query parsing: "?a=1&b=2" → {"a": "1", "b": "2"}. Hypothesis finds bugs: Empty paths, special characters (%20), IPv6 ([::1]), ports (: vs :443), double slashes. Example-based misses: Hypothesis tests 100 random URLs, finds edge cases like "http://[::1]:8080/path%20with%20spaces?a=1&b=2".',
    keyPoints: [
      'Round-trip: build_url(parse_url(url)) == url, catches data loss',
      'Scheme extraction: scheme in ["http", "https"], catches invalid schemes',
      'Port defaults: http:80, https:443, catches missing defaults',
      'Path validation: starts with /, catches formatting bugs',
      'Hypothesis finds: IPv6, special chars, edge cases example-based tests miss',
    ],
  },
];
