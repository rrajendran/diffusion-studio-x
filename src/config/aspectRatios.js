// Stable Diffusion friendly (multiples of 64, correct ratios)
export const ASPECT_RATIOS = [
  { label: '1:1',  value: '1:1',  width: 512, height: 512 },

  // Landscape
  { label: '4:3',  value: '4:3',  width: 768, height: 576 },
  { label: '16:9', value: '16:9', width: 1024, height: 576 },

  // Portrait
  { label: '3:4',  value: '3:4',  width: 576, height: 768 },
  { label: '9:16', value: '9:16', width: 576, height: 1024 },
];

export const DEFAULT_RATIO = '16:9';

export function getScaledDimensions(ratioValue, base = 512) {
  const ratio = ASPECT_RATIOS.find(r => r.value === ratioValue) ?? ASPECT_RATIOS[0];
  // Return the pre-configured dimensions directly — they are already set to
  // SD-friendly multiples of 64 and the correct pixel sizes for each ratio.
  return { width: ratio.width, height: ratio.height };
}