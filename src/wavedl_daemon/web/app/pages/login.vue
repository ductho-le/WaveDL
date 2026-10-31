<script setup lang="ts">
import { validatedNextPath } from '../lib/auth-routing';

definePageMeta({ keepalive: false });

const { state, connection } = useWorkstation();
const { keepSignedIn } = connection;
const route = useRoute();
const router = useRouter();

watch(
  () => state.value.status,
  (status, prior) => {
    if (status === 'connected' && prior !== 'connected') {
      void router.replace(validatedNextPath(route.query.next));
    } else if (status === 'disconnected' && prior === 'connected') {
      void router.replace('/login');
    }
  },
);

function disconnect(): void {
  void connection.detach();
  void router.replace('/login');
}
</script>

<template>
  <div class="flex min-h-full flex-1 items-center justify-center" data-login-page>
    <ConnectionPanel
      :state="state"
      :keep-signed-in="keepSignedIn"
      @connect="connection.connect"
      @reconnect="connection.reconnect"
      @update:keep-signed-in="connection.setKeepSignedIn"
      @detach="disconnect"
      @refresh="connection.refresh"
    />
  </div>
</template>
