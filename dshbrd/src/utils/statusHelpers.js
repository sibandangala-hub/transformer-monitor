export const fmt = t => t ? new Date(t * 1000).toLocaleString() : '--';
export const pretty = s => s ? String(s).replaceAll('_', ' ') : '--';

export const urgCls = u => ({
  NORMAL: 'normal', WARNING: 'warning', PLAN_MAINTENANCE: 'plan',
  URGENT: 'urgent', CRITICAL: 'critical', WARMING_UP: 'warm',
})[(u || '').toUpperCase()] || '';

export const stateCls = s => ({
  LOW: 'low', NORMAL: 'normal', HIGH: 'high',
})[(s || '').toUpperCase()] || '';

export const stateDesc = (name, s) => ({
  LOW:    `${name} is below the learned normal band`,
  NORMAL: `${name} is within the learned normal band`,
  HIGH:   `${name} is above the learned normal band`,
})[(s || '').toUpperCase()] || 'State unavailable';

export const healthText = h =>
  h >= 80 ? 'Healthy' :
  h >= 60 ? 'Slight degradation' :
  h >= 40 ? 'Moderate degradation' :
  h >= 20 ? 'Poor condition' : 'Critical condition';
