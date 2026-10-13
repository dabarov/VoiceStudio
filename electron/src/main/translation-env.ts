import { envWithToolPath } from './tool-path';

/** Keep provider login and OS settings, without passing the studio's secrets
 * or inherited OpenCode settings into a run that processes untrusted dialogue.
 * Tool restrictions are enforced separately by translationLaunchArgs.
 */
export function translationEnvironment(env: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv {
  const result = envWithToolPath({ env });
  for (const key of Object.keys(result)) {
    // Case-insensitive also covers Windows environment lookup semantics.
    const name = key.toUpperCase();
    if (
      name.startsWith('OMNIVOICE_') ||
      name.startsWith('VOICESTUDIO_') ||
      name.startsWith('HF_') ||
      name === 'HUGGING_FACE_HUB_TOKEN' ||
      name === 'HUGGINGFACE_HUB_TOKEN' ||
      name.startsWith('OPENCODE_CONFIG') ||
      name === 'OPENCODE_PERMISSION'
    )
      delete result[key];
  }
  return result;
}
