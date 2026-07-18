// Generates a camera-iris/aperture SVG: N overlapping blade paths arranged
// radially around a center. Used as SpendLens's signature visual motif —
// scattered inputs resolving into a focused number.
function buildApertureSVG({ size = 320, blades = 8, openness = 0.55, colorA = "#5fc9c0", colorB = "#c9a15b", id = "ap" } = {}) {
  const cx = size / 2, cy = size / 2;
  const outerR = size * 0.46;
  const innerR = outerR * openness;
  let paths = "";
  for (let i = 0; i < blades; i++) {
    const a0 = (i / blades) * Math.PI * 2;
    const a1 = ((i + 1) / blades) * Math.PI * 2;
    const bladeSweep = (Math.PI * 2 / blades) * 1.35;
    const tipAngle = a0 + bladeSweep * 0.5;
    const p1x = cx + Math.cos(a0) * innerR, p1y = cy + Math.sin(a0) * innerR;
    const p2x = cx + Math.cos(a0 + bladeSweep) * outerR, p2y = cy + Math.sin(a0 + bladeSweep) * outerR;
    const p3x = cx + Math.cos(a1) * innerR, p3y = cy + Math.sin(a1) * innerR;
    const color = i % 2 === 0 ? colorA : colorB;
    paths += `<path class="ap-blade" data-i="${i}" d="M ${cx} ${cy} L ${p1x.toFixed(2)} ${p1y.toFixed(2)} L ${p2x.toFixed(2)} ${p2y.toFixed(2)} L ${p3x.toFixed(2)} ${p3y.toFixed(2)} Z" fill="${color}" opacity="0.16" />`;
  }
  // thin ring outline
  const ring = `<circle cx="${cx}" cy="${cy}" r="${outerR}" fill="none" stroke="${colorA}" stroke-opacity="0.35" stroke-width="1"/>`;
  const innerRing = `<circle cx="${cx}" cy="${cy}" r="${innerR}" fill="none" stroke="${colorB}" stroke-opacity="0.5" stroke-width="1" stroke-dasharray="2 4"/>`;
  return `<svg class="aperture-svg" id="${id}" viewBox="0 0 ${size} ${size}" width="${size}" height="${size}" xmlns="http://www.w3.org/2000/svg">${ring}${paths}${innerRing}</svg>`;
}

// small aperture used inline (brand mark, section markers)
function buildMiniAperture({ size = 26, blades = 6, colorA = "#5fc9c0", colorB = "#c9a15b" } = {}) {
  return buildApertureSVG({ size, blades, openness: 0.42, colorA, colorB, id: "mini-" + Math.random().toString(36).slice(2, 8) });
}
