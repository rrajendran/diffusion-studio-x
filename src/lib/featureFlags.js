import { PROVIDERS, ADAPTER_FAMILIES } from '../config/features.js'

/**
 * Builds the flags array for FlagsProvider from defaultEnabled in features.js.
 * Flag names: provider.<value>  and  adapter.<key>
 */
export function getFeatureFlags() {
  return [
    ...PROVIDERS.map(p => ({ name: `provider.${p.value}`, isActive: p.defaultEnabled })),
    ...ADAPTER_FAMILIES.map(a => ({ name: `adapter.${a.key}`, isActive: a.defaultEnabled })),
  ]
}
