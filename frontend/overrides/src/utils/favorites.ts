export type FavoriteMap = Record<string, string[]>;

const storageKey = (alias: string) => `autopcr_fav_${alias}`;

export function loadFavoriteMap(alias: string): FavoriteMap {
    const stored = localStorage.getItem(storageKey(alias));
    if (!stored) return {};

    try {
        const parsed = JSON.parse(stored) as unknown;
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return {};

        return Object.fromEntries(
            Object.entries(parsed).map(([area, modules]) => [
                area,
                Array.isArray(modules)
                    ? modules.filter((module): module is string => typeof module === 'string')
                    : [],
            ]),
        );
    } catch {
        return {};
    }
}

export function saveFavoriteMap(alias: string, favorites: FavoriteMap): void {
    localStorage.setItem(storageKey(alias), JSON.stringify(favorites));
}

export function normalizeFavoritesForArea(
    favorites: FavoriteMap,
    areaKey: string,
    moduleOrder: string[],
): FavoriteMap {
    const currentModules = new Set(moduleOrder);
    const migrated = new Set<string>();

    Object.values(favorites).forEach((modules) => {
        modules.forEach((module) => {
            if (currentModules.has(module)) migrated.add(module);
        });
    });

    const normalized = Object.fromEntries(
        Object.entries(favorites).map(([area, modules]) => [
            area,
            modules.filter((module) => !currentModules.has(module)),
        ]),
    );
    normalized[areaKey] = moduleOrder.filter((module) => migrated.has(module));
    return normalized;
}

export function normalizeFavoritesForAreas(
    favorites: FavoriteMap,
    areas: { key: string; order: string[] }[],
): FavoriteMap {
    return areas.reduce(
        (normalized, area) => normalizeFavoritesForArea(normalized, area.key, area.order),
        favorites,
    );
}
