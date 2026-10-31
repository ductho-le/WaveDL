import type { InjectionKey } from 'vue'

export type SheetSide = 'top' | 'right' | 'bottom' | 'left'

export const SHEET_SIDE_KEY: InjectionKey<SheetSide> = Symbol('sheet-side')
