import type { Component } from "vue"
import type { ChartConfig } from "."
import { isClient } from "@vueuse/core"
import { h, render } from "vue"

export function componentToString(config: ChartConfig, component: Component, props?: Record<string, unknown>) {
  if (!isClient)
    return

  return (datum: unknown, x: number | Date) => {
    const data = typeof datum === "object" && datum !== null && "data" in datum ? datum.data : datum
    const vnode = h(component, { ...props, payload: data, config, x })
    const div = document.createElement("div")
    render(vnode, div)
    const content = div.innerHTML
    render(null, div)
    return content
  }
}
