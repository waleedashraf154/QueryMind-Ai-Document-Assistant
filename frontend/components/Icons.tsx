import type { SVGProps } from "react";
type IconProps = SVGProps<SVGSVGElement>;
function Icon({ children, ...props }: IconProps) { return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>{children}</svg>; }
export const SearchIcon = (props: IconProps) => <Icon {...props}><circle cx="11" cy="11" r="6.5" /><path d="m16 16 4.5 4.5" /></Icon>;
export const PlusIcon = (props: IconProps) => <Icon {...props}><path d="M12 5v14M5 12h14" /></Icon>;
export const PanelIcon = (props: IconProps) => <Icon {...props}><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M15 4v16" /></Icon>;
export const FileIcon = (props: IconProps) => <Icon {...props}><path d="M14 2.8H7a2 2 0 0 0-2 2v14.4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7.8Z" /><path d="M14 2.8v5h5M9 13h6M9 17h4" /></Icon>;
export const LayersIcon = (props: IconProps) => <Icon {...props}><path d="m12 3 8 4.4-8 4.4-8-4.4L12 3Z" /><path d="m4 12 8 4.4 8-4.4M4 16.6l8 4.4 8-4.4" /></Icon>;
export const PaperclipIcon = (props: IconProps) => <Icon {...props}><path d="m20.5 11.5-8.9 8.9a5 5 0 1 1-7.1-7.1l9.2-9.2a3.5 3.5 0 0 1 5 5l-9.2 9.2a2 2 0 0 1-2.8-2.8l8.5-8.5" /></Icon>;
export const FaceIcon = (props: IconProps) => <Icon {...props}><path d="M8 3H6a3 3 0 0 0-3 3v2M16 3h2a3 3 0 0 1 3 3v2M8 21H6a3 3 0 0 1-3-3v-2M16 21h2a3 3 0 0 0 3-3v-2M9 10h.01M15 10h.01M9.5 15c1.6 1.5 3.4 1.5 5 0" /></Icon>;
export const ArrowIcon = (props: IconProps) => <Icon {...props}><path d="M5 12h13M13 6l6 6-6 6" /></Icon>;
export const SparkleIcon = (props: IconProps) => <Icon {...props}><path d="m12 3 1.4 5.6L19 10l-5.6 1.4L12 17l-1.4-5.6L5 10l5.6-1.4L12 3ZM19 16l.7 2.3L22 19l-2.3.7L19 22l-.7-2.3L16 19l2.3-.7L19 16Z" /></Icon>;
export const TrashIcon = (props: IconProps) => <Icon {...props}><path d="M4 7h16M9 7V4h6v3m3 0-1 13H7L6 7" /><path d="M10 11v5M14 11v5" /></Icon>;
export const PinIcon = (props: IconProps) => <Icon {...props}><path d="m14 4 6 6-3 1-4 4-1 5-2-2-5-1 5-5 1-3 3-5Z" /></Icon>;
export const MicIcon = (props: IconProps) => <Icon {...props}><rect x="9" y="3" width="6" height="11" rx="3" /><path d="M5 11a7 7 0 0 0 14 0M12 18v3M9 21h6" /></Icon>;
export const VolumeIcon = (props: IconProps) => <Icon {...props}><path d="M4 10v4h4l5 4V6l-5 4H4Z" /><path d="M17 9.5a4 4 0 0 1 0 5M19.5 7a7.5 7.5 0 0 1 0 10" /></Icon>;
export const ChatIcon = (props: IconProps) => <Icon {...props}><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" /></Icon>;
