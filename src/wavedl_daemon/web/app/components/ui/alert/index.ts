import type { VariantProps } from 'class-variance-authority'
import { cva } from 'class-variance-authority'

export { default as Alert } from './Alert.vue'
export { default as AlertDescription } from './AlertDescription.vue'
export { default as AlertTitle } from './AlertTitle.vue'

export const alertVariants = cva(
  'relative w-full rounded-lg border px-4 py-3 text-sm grid has-[>svg]:grid-cols-[calc(var(--spacing)*4)_1fr] grid-cols-[0_1fr] has-[>svg]:gap-x-3 gap-y-0.5 items-start [&>svg]:size-4 [&>svg]:translate-y-0.5 [&>svg]:text-current',
  {
    variants: {
      variant: {
        default: 'bg-card text-card-foreground',
        destructive:
          'text-red-800 bg-red-50 [&>svg]:text-red-700 *:data-[slot=alert-description]:text-red-700 border border-red-200 dark:text-red-200 dark:bg-red-950 dark:[&>svg]:text-red-300 dark:*:data-[slot=alert-description]:text-red-300 dark:border-red-800',
        warning:
          'text-amber-900 bg-amber-50 [&>svg]:text-amber-600 *:data-[slot=alert-description]:text-amber-800 border border-amber-200 dark:text-amber-100 dark:bg-amber-950 dark:[&>svg]:text-amber-300 dark:*:data-[slot=alert-description]:text-amber-200 dark:border-amber-800',
        success:
          'text-green-800 bg-green-50 [&>svg]:text-green-700 *:data-[slot=alert-description]:text-green-700 border border-green-200 dark:text-green-200 dark:bg-green-950 dark:[&>svg]:text-green-300 dark:*:data-[slot=alert-description]:text-green-300 dark:border-green-800',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  },
)

export type AlertVariants = VariantProps<typeof alertVariants>
