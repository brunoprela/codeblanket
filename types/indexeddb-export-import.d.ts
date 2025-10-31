declare module 'indexeddb-export-import' {
  export function exportToJsonString(
    idbDatabase: IDBDatabase,
    successCallback?: (jsonString: string) => void,
    errorCallback?: (error: Error) => void,
  ): Promise<string>;

  export function importFromJsonString(
    idbDatabase: IDBDatabase,
    jsonString: string,
    successCallback?: () => void,
    errorCallback?: (error: Error) => void,
  ): Promise<void>;

  export function clearDatabase(
    idbDatabase: IDBDatabase,
    successCallback?: () => void,
    errorCallback?: (error: Error) => void,
  ): Promise<void>;
}
