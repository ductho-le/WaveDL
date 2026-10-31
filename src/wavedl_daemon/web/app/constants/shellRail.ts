/**
 * The rail is an icon dock in every context — a context swaps its destinations,
 * not its shape — so this column is one fixed width rather than two.
 *
 * 64px matches the shell layout's original `RAIL_WIDTH` so main-nav pages do not
 * shift; the rail itself sits at `w-14` inside it and the icons overflow the pill
 * when the dock magnifies.
 */
export const SHELL_RAIL_WIDTH = 64;
export const SHELL_RAIL_GUTTER = 12;
