import { datasetExportMessageKeys } from '../i18n/dynamic-keys';

/** Native download handoff with no claim that the browser saved the file. */
export function usePreviewExport(filename: string) {
  const exportMessage = shallowRef<
    | ''
    | 'datasets.pngExportUnavailable'
    | 'datasets.pngExportPreparing'
    | 'datasets.pngExportPreparationFailed'
    | 'datasets.pngExportRequested'
    | 'datasets.pngExportRequestFailed'
  >('');

  const exporting = shallowRef(false);
  let revision = 0;

  function resetExport() {
    revision++;
    exporting.value = false;
    exportMessage.value = '';
  }

  onBeforeUnmount(resetExport);

  function exportPng(canvas: HTMLCanvasElement | null | undefined) {
    if (exporting.value) return;

    if (!canvas) {
      exportMessage.value = datasetExportMessageKeys.unavailable;

      return;
    }

    const current = revision;
    exporting.value = true;
    exportMessage.value = datasetExportMessageKeys.preparing;

    try {
      canvas.toBlob((blob) => {
        if (current !== revision) return;
        exporting.value = false;

        if (!blob) {
          exportMessage.value = datasetExportMessageKeys.preparationFailed;

          return;
        }

        let url: string | undefined;

        try {
          url = URL.createObjectURL(blob);
          const link = document.createElement('a');
          link.href = url;
          link.download = filename;
          link.click();
          exportMessage.value = datasetExportMessageKeys.requested;
        } catch {
          exportMessage.value = datasetExportMessageKeys.requestFailed;
        } finally {
          if (url) {
            const requestedUrl = url;
            setTimeout(() => URL.revokeObjectURL(requestedUrl), 1000);
          }
        }
      });
    } catch {
      exporting.value = false;
      exportMessage.value = datasetExportMessageKeys.preparationFailed;
    }
  }

  return { exportMessage, exporting, exportPng, resetExport };
}
