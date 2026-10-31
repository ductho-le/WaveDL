import { authRedirect } from '../lib/auth-routing';

export default defineNuxtRouteMiddleware((to) => {
  const auth = useWorkstationAuth();
  const redirect = authRedirect(to.fullPath, auth.value.status, auth.value.resumePending);

  return redirect ? navigateTo(redirect, { replace: true }) : undefined;
});
