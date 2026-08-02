import React from "react";

/**
 * A stylized pair of lungs, hand-drawn as an inline SVG rather than pulled
 * from an icon library. `@fortawesome/free-brands-svg-icons` is already a
 * dependency, but it ships logo marks (GitHub, Twitter, ...), not medical
 * icons -- adding a second Font Awesome package for one glyph would be a new
 * dependency for something four paths can do.
 *
 * `currentColor` throughout, so it inherits whatever text color sizes it
 * (matching the title beside it) without a separate color prop.
 */
const ChestIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={1.6}
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
    {...props}
  >
    <path d="M12 3v6" />
    <path d="M12 9c-1.4 0-2.6.85-3.2 2.15L7.3 14" />
    <path d="M12 9c1.4 0 2.6.85 3.2 2.15L16.7 14" />
    <path d="M7.3 14c-1.8 0-3.3 1.6-3.3 3.6S5.5 21.5 7.3 21.5c1.5 0 2.7-.95 3.05-2.3l.65-2.6" />
    <path d="M16.7 14c1.8 0 3.3 1.6 3.3 3.6s-1.5 3.9-3.3 3.9c-1.5 0-2.7-.95-3.05-2.3l-.65-2.6" />
  </svg>
);

export default ChestIcon;
