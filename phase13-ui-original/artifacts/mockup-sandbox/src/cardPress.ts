// No timer, drag shortcut, or third press. Selection persists until changed.
export function cardPress(selected: string | null, key: string, legal: boolean, busy: boolean) {
  if (!legal || busy) return {selected, submit: false};
  return selected === key ? {selected: null, submit: true} : {selected: key, submit: false};
}
