/// <reference types="vite/client" />

// Allow importing .md files with ?raw suffix
declare module '*.md?raw' {
  const content: string;
  export default content;
}

// Allow importing .md files normally (returns path)
declare module '*.md' {
  const src: string;
  export default src;
}
